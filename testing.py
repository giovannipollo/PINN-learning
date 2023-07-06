from example_no_pinn import Simple_no_pinn
from example_pinn import Simple_pinn
import time

if __name__ == "__main__":
    counter = 0
    exec_time_pinn = 0
    exec_time_no_pinn = 0
    for seed in range(400,500):
        # print("-------------- PINN ---------------")
        pinn = Simple_pinn(epochs=100, seed=seed)
        start_time = time.time()        
        mean_loss_pinn = pinn.train()
        end_time = time.time()
        exec_time_pinn += end_time - start_time
        # print("-------------- NO PINN ---------------")
        no_pinn = Simple_no_pinn(epochs=100, seed=seed)
        start_time = time.time()
        mean_loss_no_pinn = no_pinn.train()
        end_time = time.time()
        exec_time_no_pinn += end_time - start_time
        if mean_loss_pinn < mean_loss_no_pinn:
            counter += 1
        
    print(counter)
    print(exec_time_pinn/100)
    print(exec_time_no_pinn/100)