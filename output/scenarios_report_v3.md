# Extended simulation results v3 (seed-base = 500000)

Replications per cell: S0_null: 2000, S1_planted: 2000, S2_N60: 1000, S2_N80: 1000, S2_N100: 1000, S2_N120: 1000, S2_N160: 1000, S3_viol6: 2000, S5_boundary: 2000.

Monte Carlo error for probabilities (95%, worst case p = 0.5): from 0.022 (cells with largest N_rep) to 0.031.

## A. Null scenario S0: Type I error rate (two-sided rejections at α = .05) and 95% Clopper–Pearson CI

```
              Test  rate         95% CI
            H1a v1 0.065 [0.055; 0.077]
            H1a v2 0.069 [0.058; 0.081]
         H1a strat 0.065 [0.054; 0.076]
    H2a (Wilcoxon) 0.049 [0.040; 0.059]
    H2b (Wilcoxon) 0.053 [0.044; 0.064]
H2a (Mann–Whitney) 0.044 [0.035; 0.054]
H2b (Mann–Whitney) 0.048 [0.039; 0.058]
       H3 (ANCOVA) 0.044 [0.035; 0.054]
```

```
H1a variant      H2 test FWER (Holm)         95% CI confirmed in hypothesized direction
         v1     Wilcoxon       0.052 [0.043; 0.063]                               0.027
         v1 Mann–Whitney       0.050 [0.040; 0.060]                               0.029
         v2     Wilcoxon       0.053 [0.044; 0.064]                               0.027
         v2 Mann–Whitney       0.051 [0.041; 0.061]                               0.029
      strat     Wilcoxon       0.053 [0.044; 0.064]                               0.027
      strat Mann–Whitney       0.051 [0.041; 0.061]                               0.029
```

Equivalence under true difference ≈ 0 (TOST power): two-sample 0.341, paired 0.389.

## B. Power under planted effects (S1), Holm adjustment, paired Wilcoxon

```
Hypothesis Holm, H1a v1 Holm, H1a v2 Holm, H1a strat Holm (v2, Mann–Whitney)
       H1a        0.927        0.946           0.952                   0.947
    H2a_EL        0.484        0.485           0.486                   0.507
    H2b_GL        0.213        0.215           0.215                   0.222
        H3        0.597        0.597           0.597                   0.597
  all four        0.098        0.099           0.100                   0.099
```

TOST power under true equivalence: two-sample 0.350, paired 0.401.

## C. Power as a function of sample size (Holm, paired Wilcoxon, H1a v2) and TOST

```
  N   H1a H2a_EL H2b_GL    H3 all four TOST two-sample TOST paired
 40 0.946  0.485  0.215 0.597    0.099           0.350       0.401
 60 0.993  0.757  0.389 0.831    0.283           0.688       0.713
 80 1.000  0.893  0.532 0.937    0.479           0.853       0.865
100 1.000  0.968  0.661 0.976    0.629           0.933       0.946
120 1.000  0.988  0.742 0.992    0.729           0.959       0.963
160 1.000  0.997  0.894 1.000    0.892           0.996       0.996
```

## D. ITT and per-protocol (Holm, paired Wilcoxon, H1a v2)

```
  scenario analysis   H1a H2a_EL H2b_GL    H3 TOST two-sample
S1_planted      itt 0.946  0.485  0.215 0.597           0.350
S1_planted       pp 0.894  0.418  0.174 0.522           0.277
  S3_viol6      itt 0.974  0.493  0.224 0.627           0.330
  S3_viol6       pp 0.789  0.256  0.106 0.355           0.080
```

## E. TOST size at equivalence boundary (S5)

True difference ≈ 0.247 (target 0.25). Proportion of equivalence confirmations: two-sample 0.040, paired 0.045 (should be ≤ 0.05).

## F0. Non-convergence and fallback calculations

```
   scenario   N v1: SE undefined strat: jackknife required jackknife failed refits frac jackknife invalid
    S0_null  40            0.095                     0.131                        0.000             0.000
 S1_planted  40            0.021                     0.033                        0.000             0.000
     S2_N60  60            0.037                     0.051                        0.000             0.000
     S2_N80  80            0.037                     0.047                        0.000             0.000
    S2_N100 100            0.054                     0.058                        0.000             0.000
    S2_N120 120            0.059                     0.069                        0.000             0.000
    S2_N160 160            0.073                     0.092                        0.000             0.000
   S3_viol6  40            0.018                     0.023                        0.000             0.000
S5_boundary  40            0.086                     0.121                        0.000             0.000
```

## F. 95% CI coverage and bias (HR coverage evaluated on replications with defined CI)

```
   scenario   N HR coverage v1    v2 strat bias ln HR v2 bias strat coverage b
    S0_null  40          0.929 0.931 0.936         0.008      0.009      0.955
 S1_planted  40          0.903 0.904 0.887         0.106      0.145      0.951
     S2_N60  60          0.901 0.905 0.887         0.063      0.093      0.956
     S2_N80  80          0.911 0.913 0.905         0.055      0.079      0.943
    S2_N100 100          0.923 0.923 0.910         0.048      0.069      0.937
    S2_N120 120          0.917 0.921 0.915         0.036      0.054      0.938
    S2_N160 160          0.937 0.938 0.926         0.031      0.045      0.945
S5_boundary  40          0.923 0.925 0.928         0.000     -0.003      0.951
```

True values: null: HR = 0.996, HR(strat) = 0.997, b = 0.083, hidden diff = -0.001; planted: HR = 2.607, HR(strat) = 2.764, b = -1.364, hidden diff = -0.002; boundary: HR = 1.002, HR(strat) = 1.006, b = 0.183, hidden diff = 0.247
