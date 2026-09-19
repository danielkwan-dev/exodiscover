"""JSON encoding for values that come out of pandas.

`/skymap` and `/discoveries` answer straight from a DataFrame, and
`to_dict(orient="records")` hands back numpy scalars rather than Python ones.
`numpy.float64` happens to subclass `float` and survives the default encoder;
`numpy.int64` does not subclass `int` and raises. `star_id` in the sky map and
`kepid`/`rank` in the candidate ranking are all int64, so without this provider
those two endpoints fail at serialisation time -- on real data, never on an
empty fixture.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from flask.json.provider import DefaultJSONProvider


def json_records(df: pd.DataFrame) -> list[dict]:
    """A DataFrame as records that are actually valid JSON.

    Two pandas facts make a plain `to_dict(orient="records")` unsafe here:

    * A missing number comes out as `float("nan")`, which the stdlib encoder
      writes as the bare token `NaN`. That is not JSON -- `JSON.parse` rejects
      it outright -- so one missing radius poisons the entire 17,000-row sky
      map for the browser.
    * `where(pd.notna(df), None)` does not reliably fix it. Whether a column
      upcasts to object and keeps the None, or stays float64 and turns it
      straight back into NaN, depends on how pandas blocked the frame. It
      therefore works on some columns and silently fails on their neighbours,
      which is why this went unnoticed: the first few hundred sky map rows are
      complete, and the gaps start later.

    Casting to object first makes None stick in every column.
    """
    finite = df.replace([np.inf, -np.inf], np.nan)
    return finite.astype(object).where(pd.notna(finite), None).to_dict(orient="records")


class NumpyJSONProvider(DefaultJSONProvider):
    """The default provider, taught the numpy scalar types pandas returns."""

    #: Response key order carries meaning in none of these payloads, and the
    #: sky map is ~17,000 records; sorting them on every request buys nothing.
    sort_keys = False

    @staticmethod
    def default(o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return DefaultJSONProvider.default(o)
