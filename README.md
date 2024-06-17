
# Domain-Generalization-Fault-Diagnosis-Benchmark

![Last update](https://img.shields.io/badge/Last%20update-20240617-brightgreen)

**Details of the benchmark can be found** [here](https://ars.els-cdn.com/content/image/1-s2.0-S0951832024000395-mmc1.pdf).

## File Description

| Index | File Name        | Description                                                                                     | Paper                                                                                      |
|:-----:|:----------------:|:-----------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------:|
| 1     | CCDG.py          | Model                                                                                           | Conditional Contrastive Domain Generalization for Fault Diagnosis                          |
| 2     | CNN-C.py         | Model                                                                                           | Learn Generalization Feature via Convolutional Neural Network: A Fault Diagnosis Scheme Toward Unseen Operating Conditions |
| 3     | DANN.py          | Model                                                                                           | Adversarial Training Among Multiple Source Domains                                         |
| 4     | DCORAL.py        | Model                                                                                           | Reduce CORAL Among Multiple Source Domains                                                 |
| 5     | DDC.py           | Model                                                                                           | Reduce MMD Among Multiple Source Domains                                                   |
| 6     | DGNIS.py         | Model                                                                                           | A Domain Generalization Network Combining Invariance and Specificity Towards Real-Time Intelligent Fault Diagnosis |
| 7     | ERM.py           | Model                                                                                           | Reduce Classification Loss                                                                 |
| 8     | IEDGNet.py       | Model                                                                                           | A Hybrid Generalization Network for Intelligent Fault Diagnosis of Rotating Machinery Under Unseen Working Conditions |
| 9     | data_loaded_1d.py| Data Preparation                                                                                | /                                                                                          |
| 10    | resnet18_1d.py   | Network                                                                                         | /                                                                                          |
| 11    | utils.py         | Metrics                                                                                         | /                                                                                          |

## Dataset Preparation


**You can find the data for the DG benchmark [here](https://pan.quark.cn/s/981b7072139a).**


### Cross-Working Condition

For example:

```python
CWRUTasksetting = {
    'dataset': 'C-CWRU', 
    'class_num': 10, 
    'src_tar': np.array([[0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 3, 1], [1, 2, 3, 0]])
}
```

In the C-CWRU dataset, there are four working conditions (0, 1, 2, 3) with 10 types of healthy vibration data. Four tasks are created by selecting one condition as the target domain and the other three as the source domains.

### Cross-Machine Condition

For example:

```python
datasetlist = [['M_CWRU', 'M_IMS', 'M_JNU', 'M_HUST']]
```

This indicates using data from CWRU, IMS, and JNU as source domains (combining all working conditions) and the HUST dataset as the target domain (also combining all working conditions).

## Contact

If you have any questions, please feel free to contact me:

- **Name:** Chao Zhao
- **Email:** zhaochao734@hust.edu.cn

## BibTeX Citation

If you find this paper and repository useful, please cite our paper 😊.

```bibtex
@article{Zhao2024domain,
  title={Domain Generalization for Cross-Domain Fault Diagnosis: an Application-oriented Perspective and a Benchmark Study},
  author={Zhao, Chao and Zio, Enrico and Shen, Weiming},
  journal={Reliability Engineering & System Safety},
  pages={109964},
  year={2024}
}
```

