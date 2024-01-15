# to run experiment for static graph
python3 honeypot/driver_competent.py --budget 10 --start 20 --fn adsim05 --algo mixed_attack

--algo:
  - mixed_attack: mip for the joint problem of competent and simple attacker
  - greedy_flat: greedy algorithm for simple attacker
  - greedy_competent: greedy algorithm for competent attacker
