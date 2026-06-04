# 2. Profiling and Benchmarking

## 2.1 Profiling

### benchmarking_script

(b)

TODO: 看看要不要再缩小一下context_length

出于显存限制，有别于Assignment要求，此处使用128的`context_length`，并只测试`small`与`medium`与`large`

| | small | medium |
| - | - | - |
| sizes | small | medium |
| forward - mean | 0.024908 | 0.072667 |
| forward - std | 0.003646 | 0.004726 |
| forward&backward - mean | 0.078229 | 0.224781 |
| forward&backward - std | 0.004895 | 0.013813 |
| full - mean | 0.141867 | 3.022274 |
| full - std | 0.010374 | 0.296010 |

标准差整体偏小

受显存限制，`medium`的`full`已经会使用到swap的显存，所以速度突然会慢很多

(c)

无warm-up：

| | small | medium |
| - | - | - |
| sizes | small | medium |
| forward - mean | 0.043032 | 0.083298 |
| forward - std | 0.048476 | 0.016569 |
| forward&backward - mean | 0.100210 | 0.245581 |
| forward&backward - std | 0.031393 | 0.016981 |
| full - mean | 0.153269 | 3.676663 |
| full - std | 0.025304 | 1.559323 |

1 warm-up：

| | small | medium |
| - | - | - |
| sizes | small | medium |
| forward - mean | 0.026975 | 0.084398 |
| forward - std | 0.002026 | 0.001589 |
| forward&backward - mean | 0.081818 | 0.243896 |
| forward&backward - std | 0.005415 | 0.018914 |
| full - mean | 0.146753 | 3.031019 |
| full - std | 0.008127 | 1.279067 |

2 warm-up：

| | small | medium |
| - | - | - |
| sizes | small | medium |
| forward - mean | 0.024448 | 0.081506 |
| forward - std | 0.001912 | 0.001729 |
| forward&backward - mean | 0.082379 | 0.246422 |
| forward&backward - std | 0.005273 | 0.010428 |
| full - mean | 0.147741 | 3.349467 |
| full - std | 0.009598 | 1.994165 |

无warm-up会让平均结果和标准差都增加，而加了一点warm-up后就会缓解

无warm-up时前几轮运行较慢，后面变快，从而影响标准差。有warm-up时启动的几轮会计入warm-up中，后续实验数值记录会更准确
