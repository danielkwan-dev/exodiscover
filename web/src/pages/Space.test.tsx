import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import Space from "./Space";

/** jsdom has no WebGL, so `createStarfield` returns null and the scene draws
 * nothing. Every call site optional-chains it, so the overlays still mount —
 * which is exactly the layer these tests are about.
 *
 * `getContext` is stubbed rather than left to jsdom's own unimplemented stub:
 * that one throws into the console on every mount, and noise in a passing run
 * is how a real error gets missed later. */
beforeEach(() => {
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
});

const skymap = {
  n: 1,
  objects: [
    {
      name: "Kepler-22 b",
      mission: "Kepler",
      star_id: 10593626,
      ra: 290.0,
      dec: 47.9,
      dist_pc: 190.0,
      disposition: "CONFIRMED",
      probability: 0.94,
      radius_earth: 2.38,
      period_days: 289.86,
      top_reasons: "koi_model_snr:+1.82",
    },
  ],
};

const transfer = {
  in_domain: {
    n: 2298,
    roc_auc: 0.9653375882935434,
    pr_auc: 0.934481155214731,
    brier: 0.06943664821989647,
    base_rate: 0.36858137510879024,
    accuracy: 0.902088772845953,
    majority_baseline: 0.6314186248912097,
    confusion_matrix: [[1408, 43], [182, 665]],
  },
  zero_shot: {
    n: 2562,
    roc_auc: 0.8376105048559666,
    pr_auc: 0.8030560959313546,
    brier: 0.17705411285134232,
    base_rate: 0.49453551912568305,
    accuracy: 0.7697111631537861,
    majority_baseline: 0.505464480874317,
    confusion_matrix: [[1020, 275], [315, 952]],
  },
  roc_auc_drop: 0.12772708343757688,
  brier_ratio: 2.54986548732358,
};

/** Routes by path so the two queries the scene issues can succeed or fail
 * independently — the point of the second test below. */
function stubApi(routes: Record<string, { ok: boolean; body: unknown }>) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      const match = Object.entries(routes).find(([path]) => url.includes(path));
      if (!match) return { ok: false, status: 404, statusText: "Not Found", json: async () => ({}) };
      const [, r] = match;
      return {
        ok: r.ok,
        status: r.ok ? 200 : 503,
        statusText: r.ok ? "OK" : "Service Unavailable",
        json: async () => r.body,
      };
    }),
  );
}

function renderScene() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <Space />
    </QueryClientProvider>,
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("Space", () => {
  it("shows the zero-shot accuracy in the scene", async () => {
    stubApi({
      "/skymap": { ok: true, body: skymap },
      "/metrics": { ok: true, body: { transfer } },
    });
    renderScene();
    expect(await screen.findByText("77.0%")).toBeInTheDocument();
    expect(screen.getByText(/baseline 50\.5%/)).toBeInTheDocument();
  });

  it("still renders the scene when /metrics is unavailable", async () => {
    // `exo train` may never have been run against this checkout. The catalogue
    // is what the app is for; the metrics panel is not worth failing over.
    stubApi({
      "/skymap": { ok: true, body: skymap },
      "/metrics": { ok: false, body: { detail: "No metrics available. Run `exo train`." } },
    });
    renderScene();
    expect(await screen.findByText(/drag to look/)).toBeInTheDocument();
    expect(screen.queryByText("77.0%")).not.toBeInTheDocument();
  });
});
