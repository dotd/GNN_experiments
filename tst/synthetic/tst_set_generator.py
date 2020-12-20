import numpy as np
import src.synthetic.set_generator as generator


def tst_generate_synthetic_sets():
    num_samples = 10
    num_centers = 3
    alpha_bet = 20
    noise = 0.05
    min_num = 3
    max_num = 7
    random = np.random.RandomState(1)

    synthetic_set = generator.generate_synthetic_sets(num_samples=num_samples,
                                                      num_centers=num_centers,
                                                      alpha_bet=alpha_bet,
                                                      noise=noise,
                                                      min_num=min_num,
                                                      max_num=max_num,
                                                      random=random)
    print(f"synthetic set:\n{synthetic_set}")


if __name__ == "__main__":
    tst_generate_synthetic_sets()
