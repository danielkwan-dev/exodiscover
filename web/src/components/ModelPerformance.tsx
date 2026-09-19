import type { Metrics } from "../lib/api";

/**
 * The one accuracy this project quotes, in the scene that demonstrates it.
 *
 * The number is zero-shot on TESS because that is the only population where an
 * accuracy carries information: it is near-balanced, so 77% against a 50.5%
 * baseline is a real 26 points of signal. The in-domain Kepler figures are
 * higher and deliberately absent — Kepler's held-out slice is 63% false
 * positives, so its baseline alone is 0.631 and the two numbers are not
 * comparable. `docs/LEAKAGE.md` §5 works this through.
 *
 * The baseline is rendered next to the accuracy rather than below it, because
 * an accuracy shown alone is the thing that gets quoted alone.
 */

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

const ModelPerformance = ({ transfer }: { transfer?: Metrics["transfer"] }) => {
  // No metrics means `exo train` has not been run against this checkout. The
  // scene still works without them, so the panel simply is not there.
  if (!transfer) return null;

  const z = transfer.zero_shot;

  return (
    <div className="w-64 rounded border border-white/10 bg-black/50 p-4 text-xs backdrop-blur-sm">
      <p className="opacity-40">model performance</p>

      <div className="mt-2 flex items-baseline gap-2">
        <span className="font-mono text-2xl text-[#ff8b45]">{pct(z.accuracy)}</span>
        <span className="opacity-50">baseline {pct(z.majority_baseline)}</span>
      </div>

      <p className="mt-1 leading-relaxed opacity-60">
        trained on Kepler, zero-shot on TESS — a telescope it never saw in training
      </p>

      <div className="mt-3 space-y-1 border-t border-white/10 pt-3">
        <Row label="ROC-AUC" value={z.roc_auc.toFixed(3)} />
        <Row label="Brier" value={z.brier.toFixed(3)} />
        <Row label="objects" value={z.n.toLocaleString()} />
      </div>
    </div>
  );
};

const Row = ({ label, value }: { label: string; value: string }) => (
  <div className="flex justify-between">
    <span className="opacity-50">{label}</span>
    <span className="font-mono">{value}</span>
  </div>
);

export default ModelPerformance;
