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

### nsys_profile

出于显存限制，我只测试了`small`参数量的模型

(a)

| Context Length | 128 | 256 | 512 |
| - | - | - | - |
| Forward Time - Python | 0.023 | 0.044 | 0.101 |
| Forward Time - nsys | 0.020 | 0.021 | 0.023 |

不match：Python测量的时间会随着context增长而明显增长，而nsys测量出的forward本身的时间增长幅度很小

(b)

对于512 context length：
- `ampere_sgemm_128x64_tn`
- 1275
- 是的。但后者的时间占用分布会均匀一些，不像前者中此项绝对领先

(c)

- `vectorized_elementwise_kernel`：逐元素操作kernel，会尽量一次处理多个连续元素
- `elementwise_kernel`：同为逐元素操作kernel，但是更泛化
- `reduce_kernel`：做reduce操作（即多个元素汇总成更少的元素）的kernel

(d)

- 其他矩阵乘法的kernel耗时占比上升，且与(b)中占比最高的kernel的占比接近
- 其他kernel的占比变数不大

(e)

| Computation | Runtime | FLOPs |
| - | - | - |
| Attention Scores | 145.93μs | 1,623,195,648 |
| Softmax | 66.91μs | 62,865,408 |
| Final Matmul | 106.99μs | 1,610,612,736 |

Runtime与FLOPs基本同大同小

### mixed_precision_accumulation

结果：

```text
tensor(10.0001)
tensor(9.9531, dtype=torch.float16)
tensor(10.0021)
tensor(10.0021)
```

精度：全FP32 > FP16使用FP32累积 > 全FP16

### benchmarking_mixed_precision

(a)

- FP32：autocast不会真的更改模型参数类型
- FP16：矩阵乘法，很适合被autocast
- FP32：包含reduce类对数值精度更敏感的内容
- FP16：ReLU一般不改变dtype
- FP32：为了稳定性，需要FP32
- FP32：与实际参数类型有关

(b)

- 在取平均值时需要做sum
- 可以不，因为其数据动态范围更大；但是精度方面依然存在局限，因此使用FP32会更好

(c)

| size | no-mixed | mixed-precision |
| - | - | - |
| small | 0.289 | 0.174 |
| medium | 