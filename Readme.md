# DeepN-JPEG


https://arxiv.org/pdf/1803.05787.pdf

## This is a simple implementation of our work "DeepN-JPEG: A Deep Neural Network Favorable JPEG-based Image Compression Framework"
* We implement frequency analysis modul for one image (can be extend to multiple images)
* You need to map the generated matrix to quantization table and replace the original one in JPEG framework. e.g.:
```bash
./cjpeg -dct int -qtable qt.txt -baseline -opt  -outfile $C temp.bmp
````
* If you use this code please cite our paper: 

```bash 
@inproceedings{liu2018deepn,
  title={DeepN-JPEG: A deep neural network favorable JPEG-based image compression framework},
  author={Liu, Zihao and Liu, Tao and Wen, Wujie and Jiang, Lei and Xu, Jie and Wang, Yanzhi and Quan, Gang},
  booktitle={Proceedings of the 55th Annual Design Automation Conference},
  pages={18},
  year={2018},
  organization={ACM}
}
```

### Example.
```bash 
python DeepN.py
```
