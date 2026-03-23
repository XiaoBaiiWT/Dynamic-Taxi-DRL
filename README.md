# DRL Assignment 1 — Dynamic Taxi Q-Learning

## Dependencies
Python 3.8+, no external packages beyond standard library.

## Training
```bash
python train.py
```
Trains for 500,000 episodes, saves `q_table.pkl`. 

## Evaluation
```bash
python custom_taxi_env.py
```

## Files
- `train.py` — Q-learning training loop
- `state.py` — state abstraction and feature encoding  
- `shaping.py` — PBRS potential function and action penalties
- `intrinsic.py` — count-based exploration bonus
- `student_agent.py` — loads Q-table, exposes `get_action(obs)`
- `q_table.pkl` — pre-trained Q-table
(https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/6uceRjti)
