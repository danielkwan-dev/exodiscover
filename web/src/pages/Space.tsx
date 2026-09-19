import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import ModelPerformance from "../components/ModelPerformance";
import { api, type SkyObject } from "../lib/api";
import { celestialToCartesian } from "../lib/projection";
import { createStarfield, type Camera, type Starfield } from "../lib/starfield";
import { transformPoint } from "../lib/mat4";

/**
 * The whole application: one scene you are inside.
 *
 * Two missions are drawn together because that combination is the point.
 * Kepler stared at a single 22x16 degree window, so its objects form a dense
 * beam in one direction. TESS surveyed the entire sky, so its objects are in
 * every direction and the nearest is 21 light years away. An observer at Earth
 * really is surrounded -- by TESS. Drawing Kepler alone and calling it
 * "surrounded" would be a lie about the survey.
 */

const LY_PER_PC = 3.261563777;

const COLOURS: Record<string, [number, number, number]> = {
  CONFIRMED: [0.96, 0.93, 0.86],
  CANDIDATE: [1.0, 0.55, 0.24],
  "FALSE POSITIVE": [0.36, 0.4, 0.5],
};

const DISPOSITIONS = ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"];
const MISSIONS = ["Kepler", "TESS"];

interface Placed {
  o: SkyObject;
  x: number;
  y: number;
  z: number;
  ly: number;
}

const Space = () => {
  const { data, isPending, error } = useQuery({
    queryKey: ["skymap"],
    queryFn: api.skymap,
    retry: false,
  });

  // Separate query, deliberately not gating the scene. A checkout that has
  // never run `exo train` has no metrics.json and gets a 503 here; the
  // catalogue is still worth showing, so this failure stays silent.
  const { data: metrics } = useQuery({
    queryKey: ["metrics"],
    queryFn: api.metrics,
    retry: false,
  });

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fieldRef = useRef<Starfield | null>(null);
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);

  // Camera state lives in a ref: the frame loop reads it sixty times a second
  // and routing that through React would re-render the scene to move a number.
  const camera = useRef<Camera>({
    position: [0, 0, 0],
    yaw: 0.9,
    pitch: -0.15,
    fovY: Math.PI / 3,
  });
  const [drifting, setDrifting] = useState(true);
  const [selected, setSelected] = useState<SkyObject | null>(null);
  const [travelLy, setTravelLy] = useState(0);

  const [maxLy, setMaxLy] = useState(6000);
  const [missions, setMissions] = useState<string[]>(MISSIONS);
  const [kinds, setKinds] = useState<string[]>(DISPOSITIONS);
  const [minProb, setMinProb] = useState(0);

  const placed = useMemo<Placed[]>(() => {
    if (!data) return [];
    return data.objects.map((o) => {
      const v = celestialToCartesian(o.ra, o.dec, o.dist_pc);
      return { o, x: v.x, y: v.y, z: v.z, ly: o.dist_pc * LY_PER_PC };
    });
  }, [data]);

  const visible = useMemo(
    () =>
      placed.filter(
        (p) =>
          p.ly <= maxLy &&
          missions.includes(p.o.mission) &&
          kinds.includes(p.o.disposition) &&
          p.o.probability >= minProb,
      ),
    [placed, maxLy, missions, kinds, minProb],
  );

  // Set up GL once, then upload buffers only when the filters change.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const field = createStarfield(canvas);
    fieldRef.current = field;
    return () => {
      field?.destroy();
      fieldRef.current = null;
    };
  }, []);

  useEffect(() => {
    const field = fieldRef.current;
    if (!field) return;
    const n = visible.length;
    const positions = new Float32Array(n * 3);
    const colours = new Float32Array(n * 3);
    const sizes = new Float32Array(n);
    visible.forEach((p, i) => {
      positions[i * 3] = p.x;
      positions[i * 3 + 1] = p.y;
      positions[i * 3 + 2] = p.z;
      const c = COLOURS[p.o.disposition] ?? [0.6, 0.6, 0.6];
      colours[i * 3] = c[0];
      colours[i * 3 + 1] = c[1];
      colours[i * 3 + 2] = c[2];
      sizes[i] = 30 + 26 * Math.min(2.2, (p.o.radius_earth ?? 2) ** 0.32);
    });
    field.setStars(positions, colours, sizes);
  }, [visible]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let frame = 0;
    let last = performance.now();

    const render = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      if (!drag.current && drifting) camera.current.yaw += 0.02 * dt;

      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const w = Math.floor(canvas.clientWidth * dpr);
      const h = Math.floor(canvas.clientHeight * dpr);
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      fieldRef.current?.draw(camera.current, w, h);
      frame = requestAnimationFrame(render);
    };
    frame = requestAnimationFrame(render);
    return () => cancelAnimationFrame(frame);
  }, [drifting]);

  /** World-space direction the camera is looking, from its yaw and pitch. */
  const forward = useCallback((): [number, number, number] => {
    const { yaw, pitch } = camera.current;
    return [
      Math.sin(yaw) * Math.cos(pitch),
      -Math.sin(pitch),
      -Math.cos(yaw) * Math.cos(pitch),
    ];
  }, []);

  /** Travel along the view direction, in steps proportional to how far out we
   *  already are -- one parsec is nothing when the field is thousands deep. */
  const travel = useCallback(
    (direction: number) => {
      const [fx, fy, fz] = forward();
      const p = camera.current.position;
      const out = Math.hypot(p[0], p[1], p[2]);
      const step = direction * Math.max(40, out * 0.12);
      p[0] += fx * step;
      p[1] += fy * step;
      p[2] += fz * step;
      setTravelLy(Math.hypot(p[0], p[1], p[2]) * LY_PER_PC);
    },
    [forward],
  );

  /** Nearest star to the click, in screen space, using the drawn matrix. */
  const pick = useCallback(
    (clientX: number, clientY: number) => {
      const canvas = canvasRef.current;
      const field = fieldRef.current;
      if (!canvas || !field) return;
      const rect = canvas.getBoundingClientRect();
      const px = clientX - rect.left;
      const py = clientY - rect.top;

      const mvp = field.viewProjection(camera.current, rect.width / rect.height);
      let best: SkyObject | null = null;
      let bestD = 18; // pixels

      for (const p of visible) {
        const c = transformPoint(mvp, [p.x, p.y, p.z, 1]);
        if (c[3] <= 0) continue; // behind the camera
        const sx = ((c[0] / c[3]) * 0.5 + 0.5) * rect.width;
        const sy = (1 - ((c[1] / c[3]) * 0.5 + 0.5)) * rect.height;
        const d = Math.hypot(sx - px, sy - py);
        if (d < bestD) {
          bestD = d;
          best = p.o;
        }
      }
      setSelected(best);
    },
    [visible],
  );

  const toggle = (list: string[], set: (v: string[]) => void, value: string) =>
    set(list.includes(value) ? list.filter((v) => v !== value) : [...list, value]);

  return (
    <div className="fixed inset-0 overflow-hidden bg-[#04060a] text-[#dfe6f2]">
      <canvas
        ref={canvasRef}
        className="absolute inset-0 h-full w-full cursor-grab active:cursor-grabbing"
        onPointerDown={(e) => {
          drag.current = { x: e.clientX, y: e.clientY, moved: false };
          e.currentTarget.setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          if (!drag.current) return;
          const dx = e.clientX - drag.current.x;
          const dy = e.clientY - drag.current.y;
          if (Math.abs(dx) + Math.abs(dy) > 3) drag.current.moved = true;
          camera.current.yaw += dx * 0.004;
          camera.current.pitch = Math.max(
            -1.5,
            Math.min(1.5, camera.current.pitch + dy * 0.004),
          );
          drag.current = { x: e.clientX, y: e.clientY, moved: drag.current.moved };
        }}
        onPointerUp={(e) => {
          const wasDrag = drag.current?.moved;
          drag.current = null;
          if (!wasDrag) pick(e.clientX, e.clientY);
        }}
        onWheel={(e) => travel(e.deltaY < 0 ? 1 : -1)}
      />

      {isPending && (
        <p className="absolute left-6 top-6 text-sm opacity-70">Loading the catalogue…</p>
      )}
      {error && (
        <p className="absolute left-6 top-6 max-w-sm text-sm text-[#ff8b45]">
          {(error as Error).message}
        </p>
      )}

      {data && (
        <>
          <div className="absolute left-6 top-6">
            <ModelPerformance transfer={metrics?.transfer} />
          </div>

          {/* Filters: the only persistent text in the scene. */}
          <div className="absolute bottom-6 left-6 w-64 space-y-4 rounded border border-white/10 bg-black/50 p-4 text-xs backdrop-blur-sm">
            <div>
              <div className="flex justify-between opacity-60">
                <span>within</span>
                <span className="font-mono">{maxLy.toLocaleString()} ly</span>
              </div>
              <input
                type="range"
                min={20}
                max={20000}
                step={20}
                value={maxLy}
                onChange={(e) => setMaxLy(Number(e.target.value))}
                className="mt-1 w-full accent-[#ff8b45]"
              />
            </div>

            <div>
              <div className="flex justify-between opacity-60">
                <span>model probability</span>
                <span className="font-mono">{minProb.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0}
                max={0.99}
                step={0.01}
                value={minProb}
                onChange={(e) => setMinProb(Number(e.target.value))}
                className="mt-1 w-full accent-[#ff8b45]"
              />
            </div>

            <div className="flex flex-wrap gap-x-3 gap-y-1">
              {MISSIONS.map((m) => (
                <label key={m} className="flex items-center gap-1.5">
                  <input
                    type="checkbox"
                    checked={missions.includes(m)}
                    onChange={() => toggle(missions, setMissions, m)}
                    className="accent-[#ff8b45]"
                  />
                  {m}
                </label>
              ))}
            </div>

            <div className="space-y-1">
              {DISPOSITIONS.map((k) => (
                <label key={k} className="flex items-center gap-1.5">
                  <input
                    type="checkbox"
                    checked={kinds.includes(k)}
                    onChange={() => toggle(kinds, setKinds, k)}
                    className="accent-[#ff8b45]"
                  />
                  <span
                    aria-hidden
                    className="inline-block h-1.5 w-1.5 rounded-full"
                    style={{
                      background: `rgb(${COLOURS[k].map((v) => v * 255).join(",")})`,
                    }}
                  />
                  <span className="lowercase opacity-80">{k}</span>
                </label>
              ))}
            </div>

            <div className="flex items-center justify-between border-t border-white/10 pt-3 opacity-60">
              <span className="font-mono">{visible.length.toLocaleString()} shown</span>
              <label className="flex items-center gap-1.5">
                <input
                  type="checkbox"
                  checked={drifting}
                  onChange={(e) => setDrifting(e.target.checked)}
                  className="accent-[#ff8b45]"
                />
                drift
              </label>
            </div>
          </div>

          <p className="absolute bottom-6 right-6 text-right font-mono text-[11px] opacity-40">
            drag to look · scroll to travel · click a planet
            <br />
            {travelLy < 1
              ? "at Earth"
              : `${travelLy.toLocaleString(undefined, { maximumFractionDigits: 0 })} ly from Earth`}
          </p>

          {selected && (
            <div className="absolute right-6 top-6 w-72 rounded border border-white/10 bg-black/60 p-4 text-xs backdrop-blur-sm">
              <div className="flex items-baseline justify-between">
                <span className="font-mono text-sm">{selected.name}</span>
                <span className="opacity-50">{selected.mission}</span>
              </div>

              <div className="mt-3 space-y-1 border-t border-white/10 pt-3">
                <Row
                  label="model says"
                  value={`${selected.probability.toFixed(2)}  ${
                    selected.probability >= 0.5 ? "planet" : "false positive"
                  }`}
                  accent
                />
                <Row label="archive says" value={selected.disposition.toLowerCase()} />
                <Row
                  label="distance"
                  value={`${(selected.dist_pc * LY_PER_PC).toLocaleString(undefined, {
                    maximumFractionDigits: 0,
                  })} ly`}
                />
                {selected.radius_earth != null && (
                  <Row label="radius" value={`${selected.radius_earth.toFixed(2)} R⊕`} />
                )}
                {selected.period_days != null && (
                  <Row label="period" value={`${selected.period_days.toFixed(2)} d`} />
                )}
              </div>

              {selected.top_reasons && (
                <div className="mt-3 border-t border-white/10 pt-3">
                  <p className="mb-1 opacity-40">why</p>
                  {selected.top_reasons.split(";").map((part) => {
                    const [name, value] = part.split(/:(?=[+-])/);
                    const positive = value?.startsWith("+");
                    return (
                      <div key={part} className="flex justify-between font-mono">
                        <span className="opacity-70">{name}</span>
                        <span className={positive ? "text-[#ff8b45]" : "opacity-50"}>
                          {value}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}

              <button
                onClick={() => setSelected(null)}
                className="mt-3 w-full border border-white/10 py-1 opacity-60 hover:opacity-100"
              >
                close
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
};

const Row = ({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) => (
  <div className="flex justify-between">
    <span className="opacity-50">{label}</span>
    <span className={`font-mono ${accent ? "text-[#ff8b45]" : ""}`}>{value}</span>
  </div>
);

export default Space;
