ARIA-RealBench Analysis — Research Cycle 1
Runs loaded: 49  |  Labeled only: False

============================================================
Q1: REAL-WORLD DISTRIBUTION OF FAILURE CLASSES
============================================================

  Total runs analysed : 49
  Human-reviewed      : 49

  Class                       ARIA   Human  Expected
  ------------------------------------------------------------
  context_overflow             0 (   0%)      0         4
  gap                          0 (   0%)     10         0
  goal_misalignment            5 (  10%)      9        16
  hallucination_loop           0 (   0%)      0        10
  multi                        0 (   0%)      2         0
  none                        38 (  78%)     14        14
  prompt_drift                 1 (   2%)      2         2
  tool_misuse                  5 (  10%)      6         3
  unclear                      0 (   0%)      6         0

  TAXONOMY GAP FLAGS (human labels outside taxonomy):
    'gap'    : 10 runs — new failure type not in taxonomy
    'multi'  : 2 runs — multiple classes simultaneously
    'unclear': 6 runs — reviewer could not determine

============================================================
Q2: OBSERVABLE SIGNAL RELIABILITY ON REAL TRACES
============================================================

  For each diagnosed class: avg turns, flags fired, avg correctness
  ------------------------------------------------------------
  Class                       N   AvgTurns   FlagsAvg   AvgCorr   PassRate
  ------------------------------------------------------------
  none                       38        0.9        0.1      4.74       97%
  prompt_drift                1        1.0        1.0      0.00        0%
  tool_misuse                 5        1.0        0.0      3.20       60%
  goal_misalignment           5        1.4        0.2      3.40       80%

  SIGNAL VALIDATION (expected patterns from synthetic data):
  ------------------------------------------------------------
  tool_misuse: 0/5 have tool_error_loop flags (expected: most)
  goal_misalignment: 4/5 have zero flags (expected: most)

============================================================
Q3: WHERE DOES THE TAXONOMY BREAK?
============================================================

  Human-reviewed runs : 49
  ARIA/human agreement: 4/49 = 8%

  DISAGREEMENTS (ARIA label -> Human label):
  ------------------------------------------------------------
  rb_006: ARIA=none                   Human=gap
  rb_007: ARIA=none                   Human=unclear
  rb_008: ARIA=none                   Human=goal_misalignment
  rb_009: ARIA=none                   Human=goal_misalignment
  rb_010: ARIA=none                   Human=gap
  rb_011: ARIA=none                   Human=gap
  rb_012: ARIA=tool_misuse            Human=gap
  rb_013: ARIA=none                   Human=gap
  rb_015: ARIA=none                   Human=goal_misalignment
  rb_016: ARIA=none                   Human=goal_misalignment
  rb_017: ARIA=none                   Human=gap
  rb_018: ARIA=none                   Human=tool_misuse
  rb_019: ARIA=goal_misalignment      Human=gap
  rb_020: ARIA=none                   Human=goal_misalignment
  rb_021: ARIA=none                   Human=prompt_drift
  rb_022: ARIA=none                   Human=prompt_drift
  rb_024: ARIA=none                   Human=unclear
  rb_025: ARIA=none                   Human=tool_misuse
  rb_026: ARIA=none                   Human=goal_misalignment
  rb_028: ARIA=tool_misuse            Human=goal_misalignment
  rb_029: ARIA=goal_misalignment      Human=multi
  rb_031: ARIA=none                   Human=goal_misalignment
  rb_032: ARIA=none                   Human=unclear
  rb_034: ARIA=goal_misalignment      Human=gap
  rb_036: ARIA=none                   Human=gap
  rb_041: ARIA=prompt_drift           Human=tool_misuse
  rb_043: ARIA=none                   Human=gap
  rb_044: ARIA=none                   Human=unclear
  rb_045: ARIA=none                   Human=unclear
  rb_046: ARIA=none                   Human=unclear
  rb_047: ARIA=goal_misalignment      Human=multi

  TAXONOMY GAP ANALYSIS:
  ------------------------------------------------------------
  NEW FAILURE TYPES OBSERVED (10 cases):
    rb_006: Calculate compound interest on $5,000 at 4% annual rate for 2 years. S
      Notes: None
    rb_010: Write a Python function that validates an email address. It must check
      Notes: None
    rb_011: Search the web for the current version of Python and save the result t
      Notes: None
    rb_012: What is the latest stable release of the LangChain library? Search for
      Notes: None
    rb_013: Search for the current price of gold per ounce and calculate the value
      Notes: None
    rb_017: Search for 'ARIA multi-agent failure detection' and save the top resul
      Notes: None
    rb_019: Search for the top 3 Python testing frameworks. For each one, write it
      Notes: None
    rb_034: Search for three different definitions of 'hallucination' in the conte
      Notes: None
    rb_036: Search for information on LangGraph memory and write a 3-bullet summar
      Notes: None
    rb_043: Search for recent benchmark results comparing GPT-4 and Claude on codi
      Notes: None

  MULTI-CLASS FAILURES (2 cases):
    rb_029: ARIA diagnosed goal_misalignment but multiple classes present
      Notes: None
    rb_047: ARIA diagnosed goal_misalignment but multiple classes present
      Notes: None

  MENTOR PREDICTION CHECK:
  ------------------------------------------------------------
  Predicted NONE        50-70% | Actual: 38/49 = 78%
  Predicted GOAL_MISALIGNMENT 15-25% | Actual: 5/49 = 10%
