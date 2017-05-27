"""
builds a2c loss function
"""

import theano.tensor as T
def get_a2c_loss_symbolic(agent,pool,reward_koeff=1,gamma=0.99):
    
    #get agent's Qvalues obtained via experience replay
    #we don't unroll scan here and propagate automatic updates
    #to speed up compilation at a cost of runtime speed
    replay = pool.experience_replay
    _,_,_,_,(logits_seq,V_seq) = agent.get_sessions(replay,experience_replay=True)
    
    
    # compute pi(a|s) and log(pi(a|s)) manually [use logsoftmax]
    # we can't guarantee that theano optimizes logsoftmax automatically since it's still in dev
    logits_flat = logits_seq.reshape([-1,logits_seq.shape[-1]])
    policy_seq = T.nnet.softmax(logits_flat).reshape(logits_seq.shape)
    logpolicy_seq = T.nnet.logsoftmax(logits_flat).reshape(logits_seq.shape)

    # get policy gradient
    from agentnet.learning import a2c
    elwise_actor_loss,elwise_critic_loss = a2c.get_elementwise_objective(policy=logpolicy_seq,
                                                                         treat_policy_as_logpolicy=True,
                                                                         state_values=V_seq[:,:,0],
                                                                         actions=replay.actions[0],
                                                                         rewards=replay.rewards*reward_koeff,
                                                                         is_alive=replay.is_alive,
                                                                         gamma_or_gammas=gamma,
                                                                         n_steps=None,
                                                                         return_separate=True)
        
    # (you can change them more or less harmlessly, this usually just makes learning faster/slower)
    # also regularize to prioritize exploration
    reg_logits = T.mean(logits_seq**2)
    reg_entropy = T.mean(T.sum(policy_seq*logpolicy_seq,axis=-1))

    #add-up loss components with magic numbers 
    loss = 0.1*elwise_actor_loss.mean() +\
           0.25*elwise_critic_loss.mean() +\
           1e-3*reg_entropy +\
           1e-3*reg_logits

    return loss


