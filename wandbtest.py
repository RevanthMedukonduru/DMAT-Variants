import wandb

wandb.init(project="test_wandb")
adv_attack_budgets = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
robustness_adv_acc_results = [20, 30, 40, 50, 60, 70]


for i, budget in enumerate(adv_attack_budgets):
    wandb.log({
        "adv_budgets": budget,
        "robustness_adv_acc": robustness_adv_acc_results[i],
    })