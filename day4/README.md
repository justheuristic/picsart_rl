## Exploration and exploitation
* [__main__] David Silver lecture on exploration and expoitation - [video](https://www.youtube.com/watch?v=sGuiWX07sKw)
* Alternative lecture by J. Schulman - [video](https://www.youtube.com/watch?v=SfCa1HQMkuw)
* Russian lecture - [video](https://yadi.sk/i/bVHmu9gt3Hi9Ym)
* BNNs
  * bayes.py file (in this folder) for quick prototypes, Theano+PyMC3 for more serious stuff - [url](http://pymc-devs.github.io/pymc3/notebooks/bayesian_neural_network_advi.html)
  * Same stuff in tensorflow - [url](http://edwardlib.org/tutorials/bayesian-neural-network)
  * A post on the matter - [url](https://ferrine.github.io/blog/2017/05/08/gelato-convolutional-mnist/)
  
## More materials 
* "Deep" version: variational information maximizing exploration - [video](https://www.youtube.com/watch?v=sRIjxxjVrnY)
  * Same topics in russian - [video](https://yadi.sk/i/_2_0yqeW3HDbcn)
* Lecture covering intrinsically motivated reinforcement learning - https://www.youtube.com/watch?v=aJI_9SoBDaQ
  * [Slides](https://yadi.sk/i/8sx42nau3HEYKg)
  * Same topics in russian - https://www.youtube.com/watch?v=WCE9hhPbCmc
  * Note to lecture above: UCB-1 is not just for bernoulli rewards, but for arbitrary r in [0,1], so you can just scale any reward to [0,1] to obtain a peace of mind. It's derived directly from Hoeffding's inequality.


## Model-based RL: Planning
* Planning by dynamic programming (D. Silver) - [video](https://www.youtube.com/watch?v=Nd1-UUMVfz4)
* Planning via tree search [videos 2-6 from CS188](https://www.youtube.com/channel/UCHBzJsIcRIVuzzHVYabikTQ)
* Our lecture:
  * Slides [part1](https://yadi.sk/i/3PM9zCP33J3ub3) (intro), part2(pomdp) - [pending]
  * [Lecture](https://yadi.sk/i/lOAUu7o13JBHFz) & [seminar](https://yadi.sk/i/bkmjEZrk3JBHGF)
* Monte-carlo tree search
  *  Udacity video on monte-carlo tree search (first part of a chain) - [video](https://www.youtube.com/watch?v=onBYsen2_eA)
  * Reminder: UCB-1 - [slides](https://www.cs.bham.ac.uk/internal/courses/robotics/lectures/ucb1.pdf)
  * Monte-carlo tree search step-by-step by J.Levine - [video](https://www.youtube.com/watch?v=UXW2yZndl7U)
  * Guide to MCTS (monte-carlo tree search) - [post](http://www.cameronius.com/research/mcts/about/index.html)
  * Another guide to MCTS - [url](https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/)
* Integrating learning and planning (D. Silver) - [video](https://www.youtube.com/watch?v=ItMutbeOHtc&t=1241s)

