import time, random, os
LOG = r"D:\SQuant_Pro\logs\paper_orders.log"
random.seed(1); ok=0; total=0
os.makedirs(os.path.dirname(LOG), exist_ok=True)
for _ in range(200):
    total+=1
    if random.random()<0.996:
        ok+=1
    time.sleep(0.005)
open(LOG,"a").write(f"ok={ok}, total={total}, rate={ok/total:.4f}\n")
print("paper quick done:", LOG)
