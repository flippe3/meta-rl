import metaworld
import random

def sample_random_tasks(ml10, kind="train"):
    envs = []
    classes = []
    tasks = []
    if kind == "test":
        classes = ml10.test_classes.items()
        tasks = ml10.test_tasks
    else: 
        classes = ml10.train_classes.items()
        tasks = ml10.train_tasks


    print(kind + ":")
    for name, env_cls in classes:
        env = env_cls()
        task = random.choice([task for task in tasks
                                if task.env_name == name])
        env.set_task(task)
        envs.append(env)
        print(task.env_name)
    return envs