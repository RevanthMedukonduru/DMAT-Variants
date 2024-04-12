import wandb

wandb.init(project="wandbtest")
list1 = [1, 2, 3, 4, 5]
list2 = [6, 7, 8, 9, 10]

wandb.log({"list1": list1, "list2": list2})
wandb.finish()