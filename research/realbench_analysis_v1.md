ARIA-RealBench Analysis — Research Cycle 1
Runs loaded: 50  |  Labeled only: False

============================================================
Q1: REAL-WORLD DISTRIBUTION OF FAILURE CLASSES
============================================================

  Total runs analysed : 50
  Human-reviewed      : 50

  Class                       ARIA   Human  Expected
  ------------------------------------------------------------
  context_overflow             1 (   2%)      1         4
  goal_misalignment           18 (  36%)     24        16
  hallucination_loop           0 (   0%)      0        10
  none                        16 (  32%)     17        14
  prompt_drift                 1 (   2%)      0         3
  tool_misuse                 14 (  28%)      8         3

  TAXONOMY GAP FLAGS (human labels outside taxonomy):
    None detected yet.

============================================================
Q2: OBSERVABLE SIGNAL RELIABILITY ON REAL TRACES
============================================================

  For each diagnosed class: avg turns, flags fired, avg correctness
  ------------------------------------------------------------
  Class                       N   AvgTurns   FlagsAvg   AvgCorr   PassRate
  ------------------------------------------------------------
  none                       16        1.0        0.1      4.88       75%
  prompt_drift                1        1.0        0.0      4.00        0%
  tool_misuse                14        1.0        0.0      3.50        0%
  context_overflow            1        5.0        4.0      5.00        0%
  goal_misalignment          18        1.1        0.0      4.00        6%

  SIGNAL VALIDATION (expected patterns from synthetic data):
  ------------------------------------------------------------
  context_overflow: 1/1 have tool_repetition flags (expected: most)
  tool_misuse: 0/14 have tool_error_loop flags (expected: most)
  goal_misalignment: 18/18 have zero flags (expected: most)

============================================================
Q3: WHERE DOES THE TAXONOMY BREAK?
============================================================

  Human-reviewed runs : 50
  ARIA/human agreement: 21/50 = 42%

  DISAGREEMENTS (ARIA label -> Human label):
  ------------------------------------------------------------
  rb_003: ARIA=tool_misuse            Human=none
  rb_007: ARIA=goal_misalignment      Human=none
  rb_010: ARIA=goal_misalignment      Human=none
  rb_011: ARIA=tool_misuse            Human=goal_misalignment
  rb_012: ARIA=tool_misuse            Human=goal_misalignment
  rb_016: ARIA=none                   Human=goal_misalignment
  rb_017: ARIA=tool_misuse            Human=goal_misalignment
  rb_018: ARIA=prompt_drift           Human=goal_misalignment
  rb_020: ARIA=goal_misalignment      Human=none
  rb_024: ARIA=goal_misalignment      Human=tool_misuse
  rb_029: ARIA=tool_misuse            Human=goal_misalignment
  rb_031: ARIA=none                   Human=goal_misalignment
  rb_036: ARIA=tool_misuse            Human=goal_misalignment
  rb_037: ARIA=goal_misalignment      Human=none
  rb_041: ARIA=tool_misuse            Human=goal_misalignment
  rb_047: ARIA=none                   Human=goal_misalignment
  rb_049: ARIA=none                   Human=goal_misalignment

  TAXONOMY GAP ANALYSIS:
  ------------------------------------------------------------
  No new failure types observed yet.
  No multi-class failures observed yet.

  MENTOR PREDICTION CHECK:
  ------------------------------------------------------------
  Predicted NONE        50-70% | Actual: 16/50 = 32%
  Predicted GOAL_MISALIGNMENT 15-25% | Actual: 18/50 = 36%
