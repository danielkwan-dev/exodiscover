import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import ModelPerformance from "./ModelPerformance";
import type { Metrics } from "../lib/api";

/** The real transfer block from docs/metrics/metrics.json, trimmed to what the
 * panel reads. Using the committed numbers keeps the test honest about what a
 * viewer actually sees. */
const transfer: Metrics["transfer"] = {
  in_domain: {
    n: 2298,
    roc_auc: 0.9653375882935434,
    pr_auc: 0.934481155214731,
    brier: 0.06943664821989647,
    base_rate: 0.36858137510879024,
    accuracy: 0.902088772845953,
    majority_baseline: 0.6314186248912097,
    confusion_matrix: [
      [1408, 43],
      [182, 665],
    ],
  },
  zero_shot: {
    n: 2562,
    roc_auc: 0.8376105048559666,
    pr_auc: 0.8030560959313546,
    brier: 0.17705411285134232,
    base_rate: 0.49453551912568305,
    accuracy: 0.7697111631537861,
    majority_baseline: 0.505464480874317,
    confusion_matrix: [
      [1020, 275],
      [315, 952],
    ],
  },
  roc_auc_drop: 0.12772708343757688,
  brier_ratio: 2.54986548732358,
};

describe("ModelPerformance", () => {
  it("headlines the zero-shot accuracy on TESS", () => {
    render(<ModelPerformance transfer={transfer} />);
    expect(screen.getByText("77.0%")).toBeInTheDocument();
  });

  it("pairs the accuracy with the majority-class baseline", () => {
    render(<ModelPerformance transfer={transfer} />);
    // Quoting 77% without the 51% it beats invites the reader to compare it
    // against a domain whose base rate is different.
    expect(screen.getByText(/baseline 50\.5%/)).toBeInTheDocument();
  });

  it("names both missions, so the number is read as cross-mission", () => {
    render(<ModelPerformance transfer={transfer} />);
    expect(screen.getByText(/trained on Kepler/i)).toBeInTheDocument();
    expect(screen.getByText(/zero-shot on TESS/i)).toBeInTheDocument();
  });

  it("shows the supporting metrics and the sample size", () => {
    render(<ModelPerformance transfer={transfer} />);
    expect(screen.getByText("0.838")).toBeInTheDocument();
    expect(screen.getByText("0.177")).toBeInTheDocument();
    expect(screen.getByText("2,562")).toBeInTheDocument();
  });

  it("never presents the in-domain Kepler figure as the headline", () => {
    render(<ModelPerformance transfer={transfer} />);
    // 90.2% in-domain accuracy and 0.965 in-domain ROC-AUC are real but are
    // measured on the mission the model trained on. Showing either here is the
    // mistake this panel exists to prevent.
    expect(screen.queryByText("90.2%")).not.toBeInTheDocument();
    expect(screen.queryByText("0.965")).not.toBeInTheDocument();
  });

  it("renders nothing when the metrics endpoint gave us nothing", () => {
    const { container } = render(<ModelPerformance transfer={undefined} />);
    // The scene must survive `exo train` never having been run. A missing
    // metrics file is a 503 on /metrics, not a reason to break the view.
    expect(container).toBeEmptyDOMElement();
  });
});
