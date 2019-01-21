# Deep Reinforcement Learning for robotic manuplation

Tensorflow implementation of DQN for our simulation

**A WIP Repo**

This implementation contains:

1. Deep Q-network and Q-learning
2. Experience replay memory
    - to reduce the correlations between consecutive updates
3. Network for Q-learning targets are fixed for intervals
    - to reduce the correlations between target and predicted Q-values
4. Double DQN / Duel DQN (unfinished)

## Environments

- Python 3.6
- [tqdm](https://github.com/tqdm/tqdm)
- [SciPy](http://www.scipy.org/install.html) or [OpenCV2](http://opencv.org/)
- [TensorFlow 1.9.0](https://github.com/tensorflow/tensorflow/tree/r0.12)


## Usage

First, install prerequisites with:

    $ pip install tqdm

To train a model:

    $ python main.py --is_train=True

To test the model:

    $ python main.py --is_train=False


## References

- [Source code](https://github.com/devsisters/DQN-tensorflow)


## License

MIT License.
