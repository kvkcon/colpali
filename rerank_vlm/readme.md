## data gen

- target *pipeline2VecDB*

  

k=25 colpali

#### Experience

> qs_batch: 128 * 128 * 4 bytes ≈ 65.5 KB
>
> ps_batch: 128 * 100 * 128 * 4 bytes ≈ 6.25 MB
>
> Intermediate tensor: 128 * 128 * 100 * 128 * 4 bytes ≈ 786.4 MB

 **118000/128 ≈ 922 iterations !**

> `scores` tensor  50 * 118000 * 4 bytes ≈ 22.5 MB
>
> `image_list` tensor  118000 * 100 * 128 * 2 bytes ≈ 2.81 GB



| model name          | single img embedding shape | single q embedding shape |
| ------------------- | -------------------------- | ------------------------ |
| vidore/colpali-v1.2 | 1030*128                   | n(≈40)*128               |

int8 perform great √

![image-20241016162023027](C:\Users\coconerd\AppData\Roaming\Typora\typora-user-images\image-20241016162023027.png)



### test500 

| name        | scare(row) | index  | score  | mem  |
| ----------- | ---------- | ------ | ------ | ---- |
| test        | 500        | 01m09s | 03m34s | <15G |
| train(240x) | 118k       | 4h36m  | 3360h  |      |



process   01m09s+03m34s    <15G

![image-20241016162317970](C:\Users\coconerd\AppData\Roaming\Typora\typora-user-images\image-20241016162317970.png)



