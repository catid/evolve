# Checkpoint Dynamics Report

- episodes per mode: `32`
- run count: `2`

## Best Checkpoints

| Variant | Best Greedy Update | Best Greedy Success | Best Sampled Update | Best Sampled Success | Greedy Checkpoint Exists |
| --- | ---: | ---: | ---: | ---: | --- |
| sare | 10 | 0.0000 | 20 | 0.5000 | no |
| single_expert | 10 | 0.0000 | 10 | 0.1562 | no |

## Interpretation

- `sare` never shows a nonzero greedy checkpoint in the archived series, even though its best sampled checkpoint at update `20` reaches sampled success `0.5000`.
- At `sare`'s best sampled checkpoint, the corresponding greedy success is `0.0000`, eval max-prob is `0.6671`, and eval margin is `1.5980`.
- `single_expert` never shows a nonzero greedy checkpoint in the archived series, even though its best sampled checkpoint at update `10` reaches sampled success `0.1562`.
- At `single_expert`'s best sampled checkpoint, the corresponding greedy success is `0.0000`, eval max-prob is `0.2349`, and eval margin is `0.0498`.

## Checkpoint Table

| Variant | Update | Mode | Eval Success | Eval Return | Eval Max Prob | Eval Margin | Eval Greedy Match | Train Return | Train Success | Throughput | Checkpoint |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sare | 10 | greedy | 0.0000 | 0.0000 | 0.3535 | 0.2221 | 1.0000 | 0.0000 | 0.0000 | 837.7106 | `checkpoint_update_0010.pt` |
| sare | 10 | sampled_t0.5 | 0.3125 | 0.1701 | 0.5277 | 0.2253 | 0.5198 | 0.0000 | 0.0000 | 837.7106 | `checkpoint_update_0010.pt` |
| sare | 10 | sampled_t0.7 | 0.2812 | 0.1515 | 0.4403 | 0.2281 | 0.4322 | 0.0000 | 0.0000 | 837.7106 | `checkpoint_update_0010.pt` |
| sare | 10 | sampled_t1.0 | 0.4688 | 0.2244 | 0.3575 | 0.2359 | 0.3510 | 0.0000 | 0.0000 | 837.7106 | `checkpoint_update_0010.pt` |
| sare | 20 | greedy | 0.0000 | 0.0000 | 0.6671 | 1.5980 | 1.0000 | 0.3715 | 0.6667 | 840.1288 | `checkpoint_update_0020.pt` |
| sare | 20 | sampled_t0.5 | 0.0938 | 0.0509 | 0.9381 | 1.6069 | 0.9373 | 0.3715 | 0.6667 | 840.1288 | `checkpoint_update_0020.pt` |
| sare | 20 | sampled_t0.7 | 0.4062 | 0.1973 | 0.8340 | 1.6160 | 0.8312 | 0.3715 | 0.6667 | 840.1288 | `checkpoint_update_0020.pt` |
| sare | 20 | sampled_t1.0 | 0.5000 | 0.2983 | 0.6701 | 1.6174 | 0.6601 | 0.3715 | 0.6667 | 840.1288 | `checkpoint_update_0020.pt` |
| sare | 30 | greedy | 0.0000 | 0.0000 | 0.8405 | 2.2414 | 1.0000 | 0.1454 | 0.5000 | 832.5642 | `checkpoint_update_0030.pt` |
| sare | 30 | sampled_t0.5 | 0.0000 | 0.0000 | 0.9867 | 2.2683 | 0.9840 | 0.1454 | 0.5000 | 832.5642 | `checkpoint_update_0030.pt` |
| sare | 30 | sampled_t0.7 | 0.0312 | 0.0161 | 0.9451 | 2.2604 | 0.9463 | 0.1454 | 0.5000 | 832.5642 | `checkpoint_update_0030.pt` |
| sare | 30 | sampled_t1.0 | 0.3750 | 0.1632 | 0.8422 | 2.2602 | 0.8371 | 0.1454 | 0.5000 | 832.5642 | `checkpoint_update_0030.pt` |
| sare | 40 | greedy | 0.0000 | 0.0000 | 0.9174 | 2.7058 | 1.0000 | 0.0000 | 0.0000 | 832.7463 | `checkpoint_update_0040.pt` |
| sare | 40 | sampled_t0.5 | 0.0000 | 0.0000 | 0.9956 | 2.7750 | 0.9945 | 0.0000 | 0.0000 | 832.7463 | `checkpoint_update_0040.pt` |
| sare | 40 | sampled_t0.7 | 0.0000 | 0.0000 | 0.9779 | 2.7679 | 0.9764 | 0.0000 | 0.0000 | 832.7463 | `checkpoint_update_0040.pt` |
| sare | 40 | sampled_t1.0 | 0.0000 | 0.0000 | 0.9201 | 2.7478 | 0.9184 | 0.0000 | 0.0000 | 832.7463 | `checkpoint_update_0040.pt` |
| sare | 50 | greedy | 0.0000 | 0.0000 | 0.8843 | 2.7371 | 1.0000 | 0.0000 | 0.0000 | 834.1361 | `checkpoint_update_0050.pt` |
| sare | 50 | sampled_t0.5 | 0.0000 | 0.0000 | 0.9937 | 2.7835 | 0.9923 | 0.0000 | 0.0000 | 834.1361 | `checkpoint_update_0050.pt` |
| sare | 50 | sampled_t0.7 | 0.0000 | 0.0000 | 0.9673 | 2.7815 | 0.9647 | 0.0000 | 0.0000 | 834.1361 | `checkpoint_update_0050.pt` |
| sare | 50 | sampled_t1.0 | 0.0938 | 0.0205 | 0.8856 | 2.7606 | 0.8866 | 0.0000 | 0.0000 | 834.1361 | `checkpoint_update_0050.pt` |
| sare | 60 | greedy | 0.0000 | 0.0000 | 0.8066 | 2.1700 | 1.0000 | 0.0000 | 0.0000 | 833.4327 | `checkpoint_update_0060.pt` |
| sare | 60 | sampled_t0.5 | 0.0000 | 0.0000 | 0.9776 | 2.1776 | 0.9754 | 0.0000 | 0.0000 | 833.4327 | `checkpoint_update_0060.pt` |
| sare | 60 | sampled_t0.7 | 0.0312 | 0.0171 | 0.9211 | 2.1729 | 0.9218 | 0.0000 | 0.0000 | 833.4327 | `checkpoint_update_0060.pt` |
| sare | 60 | sampled_t1.0 | 0.2812 | 0.0965 | 0.8074 | 2.1635 | 0.8049 | 0.0000 | 0.0000 | 833.4327 | `checkpoint_update_0060.pt` |
| sare | 70 | greedy | 0.0000 | 0.0000 | 0.8364 | 2.1379 | 1.0000 | 0.0000 | 0.0000 | 826.7444 | `checkpoint_update_0070.pt` |
| sare | 70 | sampled_t0.5 | 0.0000 | 0.0000 | 0.9830 | 2.1355 | 0.9817 | 0.0000 | 0.0000 | 826.7444 | `checkpoint_update_0070.pt` |
| sare | 70 | sampled_t0.7 | 0.0000 | 0.0000 | 0.9383 | 2.1386 | 0.9353 | 0.0000 | 0.0000 | 826.7444 | `checkpoint_update_0070.pt` |
| sare | 70 | sampled_t1.0 | 0.1875 | 0.0819 | 0.8365 | 2.1377 | 0.8343 | 0.0000 | 0.0000 | 826.7444 | `checkpoint_update_0070.pt` |
| sare | 80 | greedy | 0.0000 | 0.0000 | 0.8374 | 2.2396 | 1.0000 | 0.0000 | 0.0000 | 826.2175 | `checkpoint_update_0080.pt` |
| sare | 80 | sampled_t0.5 | 0.0000 | 0.0000 | 0.9847 | 2.2382 | 0.9837 | 0.0000 | 0.0000 | 826.2175 | `checkpoint_update_0080.pt` |
| sare | 80 | sampled_t0.7 | 0.0312 | 0.0056 | 0.9410 | 2.2426 | 0.9393 | 0.0000 | 0.0000 | 826.2175 | `checkpoint_update_0080.pt` |
| sare | 80 | sampled_t1.0 | 0.1875 | 0.0667 | 0.8379 | 2.2399 | 0.8349 | 0.0000 | 0.0000 | 826.2175 | `checkpoint_update_0080.pt` |
| single_expert | 10 | greedy | 0.0000 | 0.0000 | 0.2349 | 0.0498 | 1.0000 | 0.0000 | 0.0000 | 1373.9321 | `checkpoint_update_0010.pt` |
| single_expert | 10 | sampled_t0.5 | 0.0625 | 0.0277 | 0.3191 | 0.0511 | 0.3162 | 0.0000 | 0.0000 | 1373.9321 | `checkpoint_update_0010.pt` |
| single_expert | 10 | sampled_t0.7 | 0.0625 | 0.0228 | 0.2730 | 0.0495 | 0.2749 | 0.0000 | 0.0000 | 1373.9321 | `checkpoint_update_0010.pt` |
| single_expert | 10 | sampled_t1.0 | 0.1562 | 0.0677 | 0.2352 | 0.0506 | 0.2359 | 0.0000 | 0.0000 | 1373.9321 | `checkpoint_update_0010.pt` |
| single_expert | 20 | greedy | 0.0000 | 0.0000 | 0.3152 | 0.2587 | 1.0000 | 0.0000 | 0.0000 | 1382.3628 | `checkpoint_update_0020.pt` |
| single_expert | 20 | sampled_t0.5 | 0.0000 | 0.0000 | 0.4633 | 0.2640 | 0.4637 | 0.0000 | 0.0000 | 1382.3628 | `checkpoint_update_0020.pt` |
| single_expert | 20 | sampled_t0.7 | 0.0000 | 0.0000 | 0.3834 | 0.2637 | 0.3821 | 0.0000 | 0.0000 | 1382.3628 | `checkpoint_update_0020.pt` |
| single_expert | 20 | sampled_t1.0 | 0.0938 | 0.0133 | 0.3161 | 0.2672 | 0.3121 | 0.0000 | 0.0000 | 1382.3628 | `checkpoint_update_0020.pt` |
| single_expert | 30 | greedy | 0.0000 | 0.0000 | 0.4856 | 0.9740 | 1.0000 | 0.0000 | 0.0000 | 1384.7313 | `checkpoint_update_0030.pt` |
| single_expert | 30 | sampled_t0.5 | 0.0312 | 0.0075 | 0.7923 | 0.9727 | 0.7898 | 0.0000 | 0.0000 | 1384.7313 | `checkpoint_update_0030.pt` |
| single_expert | 30 | sampled_t0.7 | 0.1250 | 0.0354 | 0.6398 | 0.9731 | 0.6415 | 0.0000 | 0.0000 | 1384.7313 | `checkpoint_update_0030.pt` |
| single_expert | 30 | sampled_t1.0 | 0.0625 | 0.0190 | 0.4858 | 0.9729 | 0.4796 | 0.0000 | 0.0000 | 1384.7313 | `checkpoint_update_0030.pt` |
| single_expert | 40 | greedy | 0.0000 | 0.0000 | 0.5227 | 1.1393 | 1.0000 | 0.0000 | 0.0000 | 1384.2154 | `checkpoint_update_0040.pt` |
| single_expert | 40 | sampled_t0.5 | 0.0000 | 0.0000 | 0.8288 | 1.1395 | 0.8276 | 0.0000 | 0.0000 | 1384.2154 | `checkpoint_update_0040.pt` |
| single_expert | 40 | sampled_t0.7 | 0.0625 | 0.0310 | 0.6799 | 1.1396 | 0.6727 | 0.0000 | 0.0000 | 1384.2154 | `checkpoint_update_0040.pt` |
| single_expert | 40 | sampled_t1.0 | 0.0625 | 0.0139 | 0.5222 | 1.1376 | 0.5210 | 0.0000 | 0.0000 | 1384.2154 | `checkpoint_update_0040.pt` |
| single_expert | 50 | greedy | 0.0000 | 0.0000 | 0.5050 | 1.1113 | 1.0000 | 0.0000 | 0.0000 | 1384.0116 | `checkpoint_update_0050.pt` |
| single_expert | 50 | sampled_t0.5 | 0.0000 | 0.0000 | 0.8108 | 1.1126 | 0.8123 | 0.0000 | 0.0000 | 1384.0116 | `checkpoint_update_0050.pt` |
| single_expert | 50 | sampled_t0.7 | 0.0000 | 0.0000 | 0.6579 | 1.1112 | 0.6550 | 0.0000 | 0.0000 | 1384.0116 | `checkpoint_update_0050.pt` |
| single_expert | 50 | sampled_t1.0 | 0.0312 | 0.0199 | 0.5046 | 1.1103 | 0.5073 | 0.0000 | 0.0000 | 1384.0116 | `checkpoint_update_0050.pt` |
| single_expert | 60 | greedy | 0.0000 | 0.0000 | 0.3629 | 0.1971 | 1.0000 | 0.0000 | 0.0000 | 1386.0955 | `checkpoint_update_0060.pt` |
| single_expert | 60 | sampled_t0.5 | 0.0000 | 0.0000 | 0.5045 | 0.1912 | 0.5021 | 0.0000 | 0.0000 | 1386.0955 | `checkpoint_update_0060.pt` |
| single_expert | 60 | sampled_t0.7 | 0.0000 | 0.0000 | 0.4321 | 0.1945 | 0.4315 | 0.0000 | 0.0000 | 1386.0955 | `checkpoint_update_0060.pt` |
| single_expert | 60 | sampled_t1.0 | 0.0625 | 0.0133 | 0.3626 | 0.1936 | 0.3643 | 0.0000 | 0.0000 | 1386.0955 | `checkpoint_update_0060.pt` |
| single_expert | 70 | greedy | 0.0000 | 0.0000 | 0.4076 | 0.0996 | 1.0000 | 0.0000 | 0.0000 | 1386.8975 | `checkpoint_update_0070.pt` |
| single_expert | 70 | sampled_t0.5 | 0.0000 | 0.0000 | 0.5187 | 0.0973 | 0.5164 | 0.0000 | 0.0000 | 1386.8975 | `checkpoint_update_0070.pt` |
| single_expert | 70 | sampled_t0.7 | 0.0000 | 0.0000 | 0.4690 | 0.0957 | 0.4667 | 0.0000 | 0.0000 | 1386.8975 | `checkpoint_update_0070.pt` |
| single_expert | 70 | sampled_t1.0 | 0.0000 | 0.0000 | 0.4073 | 0.0969 | 0.4045 | 0.0000 | 0.0000 | 1386.8975 | `checkpoint_update_0070.pt` |
| single_expert | 80 | greedy | 0.0000 | 0.0000 | 0.4430 | 0.2716 | 1.0000 | 0.0000 | 0.0000 | 1389.0492 | `checkpoint_update_0080.pt` |
| single_expert | 80 | sampled_t0.5 | 0.0000 | 0.0000 | 0.6004 | 0.2699 | 0.5962 | 0.0000 | 0.0000 | 1389.0492 | `checkpoint_update_0080.pt` |
| single_expert | 80 | sampled_t0.7 | 0.0312 | 0.0038 | 0.5254 | 0.2686 | 0.5210 | 0.0000 | 0.0000 | 1389.0492 | `checkpoint_update_0080.pt` |
| single_expert | 80 | sampled_t1.0 | 0.0000 | 0.0000 | 0.4429 | 0.2693 | 0.4411 | 0.0000 | 0.0000 | 1389.0492 | `checkpoint_update_0080.pt` |
