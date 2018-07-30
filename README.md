# Incremental Label Propagation
This repository provides the implementation of our paper "Incremental Semi-Supervised Learning from Streams for Object Classification" (Ioannis Chiotellis*, Franziska Zimmermann*, Daniel Cremers and Rudolph Triebel, IROS 2018). All results presented in our work were produced with this code.

* [Installing](#usage)
* [Datasets](#data)
* [Experiments](#experiments)
* [Publication](#paper)
* [License and Contact](#other)


## <a name="usage">Installation</a>
The code was developed in python 3.5 under Ubuntu 16.04. You can clone the repo with:
```
git clone https://github.com/johny-c/incremental-label-propagation.git
```

## <a name="data">Datasets</a>
* KITTI

The repository includes 64-dimentional features extracted from KITTI sequences compressed in a zip file (data/kitti_features.zip). The included files will be extracted automatically if one of the included experiments is run on KITTI.

* MNIST

A script will automatically download the MNIST dataset if an experiment is run on it.


## <a name="experiments">Experiments</a>

The repository includes scripts that replicate the experiments found in the paper, including:

* Varying the number of labeled points or the ratio of labeled points in the data.
* Varying the number of labeled or unlabeled neighbors considered for each node.
* Varying the hyperparameter $$\theta$$ that controls the propagation area size.

To run an experiment with varying $$\theta$$:
    
    python ilp/experiments/var_theta.py -d mnist

You can set different experiment options in the .yaml files found in the experimens/cfg directory.

#### Warning:
The included experiment scripts produce a lot of statistics during the runs and the resulting files can get very large.


## <a name="paper">Publication</a>
If you use this code in your work, please cite the following paper.

Ioannis Chiotellis*, Franziska Zimmermann*, Daniel Cremers and Rudolph Triebel, _"Incremental Semi-Supervised Learning from Streams for Object Classification"_, in proceedings of the 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2018). ([pdf](https://vision.in.tum.de/_media/spezial/bib/chiotellis2018ilp.pdf))
    
*equal contribution
    
    @InProceedings{chiotellis2018incremental,
      author = "I. Chiotellis and F. Zimmermann and D. Cremers and R. Triebel",
      title = "Incremental Semi-Supervised Learning from Streams for Object Classification",
      booktitle = iros,
      year = "2018",
      month = "October",
      keywords={stream-based learning, sequential data, semi-supervised learning, object classification},
      note = {{<a href="https://github.com/johny-c/incremental-label-propagation" target="_blank">[code]</a>} },
    }

## <a name="others"> License and Contact</a>

This work is released under the [MIT Licence].

Contact **John Chiotellis** [:envelope:](mailto:chiotell@in.tum.de) for questions, comments and reporting bugs.
