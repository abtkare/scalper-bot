"""Price board: Yahoo parsing, bot live quotes preferred, delayed labels, errors never raise."""
import time

import prices

SPARK = {"spark": {"result": [
    {"symbol": "GC=F", "response": [{"meta": {"regularMarketPrice": 4050.0, "chartPreviousClose": 4000.0,
                                              "regularMarketTime": 0}}]},
    {"symbol": "CL=F", "response": [{"meta": {"regularMarketPrice": 100.0, "chartPreviousClose": 98.0,
                                              "regularMarketTime": 0}}]},
    {"symbol": "BAD", "response": []}]}}


def test_board():
    now = time.time()
    SPARK["spark"]["result"][0]["response"][0]["meta"]["regularMarketTime"] = now - 600
    SPARK["spark"]["result"][1]["response"][0]["meta"]["regularMarketTime"] = now - 30
    b = prices.PriceBoard(lambda: {"XAU/USD": (4030.5, now - 5)}, fetch=lambda syms: prices.parse_spark(SPARK))
    b.refresh()
    it = {i["key"]: i for i in b.board(now)["items"]}
    assert it["XAU"]["price"] == 4030.5 and it["XAU"]["src"] == "live" and it["XAU"]["chg_pct"] == 1.25
    assert it["WTI"]["src"] == "live" and it["WTI"]["chg_pct"] == 2.04
    assert "XAG" not in it                                      # no data -> not shown
    stale = prices.PriceBoard(lambda: {"XAU/USD": (4030.5, now - 500)}, fetch=lambda s: prices.parse_spark(SPARK))
    stale.refresh()
    assert {i["key"]: i for i in stale.board(now)["items"]}["XAU"]["src"] == "delayed"


def test_errors():
    b = prices.PriceBoard(lambda: 1 / 0, fetch=lambda s: (_ for _ in ()).throw(OSError("x")))
    b.refresh()
    assert b.err and b.board()["items"] == []


if __name__ == "__main__":
    test_board()
    test_errors()
    print("prices tests OK")
