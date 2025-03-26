# 🎲 MentalRiskES - Tasks

## Overview
All tasks focus on detecting mental disorders in users based on their comments posted on Telegram and Twitch. Given a history of messages from a user, the goal is to determine whether the user suffers from the disorder and identify the context influencing their mental health problem.

## Tasks
### Task 1: Risk Detection of Gambling Disorders
**Binary Classification**
This task aims to classify users into two categories:
- **High Risk (label = 1)**: Users showing signs of a gambling-related disorder.
- **Low Risk (label = 0)**: Users with minimal signs of gambling-related issues.

The objective is to enable early detection and facilitate timely interventions.

### Task 2: Type of Addiction Detection
**Multiclass Classification**
This task expands on Task 1 by identifying the specific type of gambling addiction a user exhibits. Every user is at some level of risk, and the model should classify them into one of the following categories:
- **Betting**: Gambling on sports-related events to win money.
- **Online Gaming**: Participation in traditional games of chance like roulette, blackjack, or slot machines.
- **Trading & Crypto**: Speculative investments in cryptocurrencies or financial trading with gambling-like behavior.
- **Lootboxes**: Purchasing randomized virtual items in video games using real money.

## Example Messages
### **Betting**
*User1 (messages):*
> "yo de frees no hable"
> 
> "pues de pago si hay buenos, pero a mi me ha llevado mi tiempo emplea tu el tuyo"

### **Online Gaming**
*User2 (messages):*
> "24 céntimos con los free spins, Roma no se construyó en un día"
> 
> "pues de pago si hay buenos, pero a mi me ha llevado mi tiempo emplea tu el tuyo"

### **Trading & Crypto**
*User3 (messages):*
> "Siii fue cuestión de rapidez porque subió a 60 y rápido se desinfló"
> 
> "interesante"

### **Lootboxes**
*User4 (messages):*
> "y al que le toco sirve para intercambio"
> 
> "para otra cosa no"

## Notes
- Whether a user is classified as high or low risk, they are always associated with a specific gambling problem.
- The addiction type considered for evaluation will be based on the latest prediction received in the final round.
- Each user belongs to exactly **one** category.
