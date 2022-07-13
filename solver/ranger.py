import math
import torch
from torch.optim.optimizer import Optimizer, required
import itertools as it

class Ranger(Optimizer):

    #Ranger deep learning optimizer - RAdam + Lookahead combined.
    #https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer

    #Ranger has now been used to capture 12 records on the FastAI leaderboard.

    #This version = 9.3.19  

    #Credits:
    #RAdam -->  https://github.com/LiyuanLucasLiu/RAdam
    #Lookahead --> rewritten by lessw2020, but big thanks to Github @LonePatient and @RWightman for ideas from their code.
    #Lookahead paper --> MZhang,G Hinton  https://arxiv.org/abs/1907.08610

    #summary of changes: 
    #full code integration with all updates at param level instead of group, moves slow weights into state dict (from generic weights), 
    #supports group learning rates (thanks @SHolderbach), fixes sporadic load from saved model issues.
    #changes 8/31/19 - fix references to *self*.N_sma_threshold; 
    #changed eps to 1e-5 as better default than 1e-8.

    def __init__(self, params, lr=1e-3, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95,0.999), eps=1e-5, weight_decay=0):
        #parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')

        #parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        #N_sma_threshold of 5 seems better in testing than 4.
        #In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.

        #prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas, N_sma_threshhold=N_sma_threshhold, eps=eps, weight_decay=weight_decay)
        super().__init__(params,defaults)

        #adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        #now we can get to work...
        #removed as we now use step from RAdam...no need for duplicate step counting
        #for group in self.param_groups:
        #    group["step_counter"] = 0
            #print("group step counter init")

        #look ahead params
        self.alpha = alpha
        self.k = k 

        #radam buffer for state
        self.radam_buffer = [[None,None,None] for ind in range(10)]

        #self.first_run_check=0

        #lookahead weights
        #9/2/19 - lookahead param tensors have been moved to state storage.  
        #This should resolve issues with load/save where weights were left in GPU memory from first load, slowing down future runs.

        #self.slow_weights = [[p.clone().detach() for p in group['params']]
        #                     for group in self.param_groups]

        #don't use grad for lookahead weights
        #for w in it.chain(*self.slow_weights):
        #    w.requires_grad = False

    def __setstate__(self, state):
        print("set state called")
        super(Ranger, self).__setstate__(state)


    def step(self, closure=None):
        loss = None
        #note - below is commented out b/c I have other work that passes back the loss as a float, and thus not a callable closure.  
        #Uncomment if you need to use the actual closure...

        #if closure is not None:
            #loss = closure()

        #Evaluate averages and grad, update param tensors
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Ranger optimizer does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]  #get state dict for this param

                if len(state) == 0:   #if first time to run...init dictionary with our desired entries
                    #if self.first_run_check==0:
                        #self.first_run_check=1
                        #print("Initializing slow buffer...should not see this at load from saved model!")
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                    #look ahead weight storage now in state dict 
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)

                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                #begin computations 
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                #compute variance mov avg
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                #compute mean moving avg
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1


                buffered = self.radam_buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                p.data.copy_(p_data_fp32)

                #integrated look ahead...
                #we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    slow_p = state['slow_buffer'] #get access to slow param tensor
                    slow_p.add_(self.alpha, p.data - slow_p)  #(fast weights - slow weights) * alpha
                    p.data.copy_(slow_p)  #copy interpolated weights to RAdam param tensor

        return loss

class RangerVA(Optimizer):

    def __init__(self, params, lr=1e-3, alpha=0.5, k=6, n_sma_threshhold=5, betas=(.95,0.999), 
                 eps=1e-5, weight_decay=0, amsgrad=True, transformer='softplus', smooth=50,
                 grad_transformer='square'):
        #parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')

        #parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        #N_sma_threshold of 5 seems better in testing than 4.
        #In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.

        #prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas, 
                        n_sma_threshhold=n_sma_threshhold, eps=eps, weight_decay=weight_decay,
                        smooth=smooth, transformer=transformer, grad_transformer=grad_transformer,
                       amsgrad=amsgrad)
        super().__init__(params,defaults)

        #adjustable threshold
        self.n_sma_threshhold = n_sma_threshhold   

        #look ahead params
        self.alpha = alpha
        self.k = k 

        #radam buffer for state
        self.radam_buffer = [[None,None,None] for ind in range(10)]

        #self.first_run_check=0

        #lookahead weights
        #9/2/19 - lookahead param tensors have been moved to state storage.  
        #This should resolve issues with load/save where weights were left in GPU memory from first load, slowing down future runs.

        #self.slow_weights = [[p.clone().detach() for p in group['params']]
        #                     for group in self.param_groups]

        #don't use grad for lookahead weights
        #for w in it.chain(*self.slow_weights):
        #    w.requires_grad = False

    def __setstate__(self, state):
        print("set state called")
        super(Ranger, self).__setstate__(state)


    def step(self, closure=None):
        loss = None
        #note - below is commented out b/c I have other work that passes back the loss as a float, and thus not a callable closure.  
        #Uncomment if you need to use the actual closure...

        #if closure is not None:
            #loss = closure()

        #Evaluate averages and grad, update param tensors
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Ranger optimizer does not support sparse gradients')
                
                amsgrad = group['amsgrad']
                smooth = group['smooth']
                grad_transformer = group['grad_transformer']

                p_data_fp32 = p.data.float()

                state = self.state[p]  #get state dict for this param

                if len(state) == 0:   #if first time to run...init dictionary with our desired entries
                    #if self.first_run_check==0:
                        #self.first_run_check=1
                        #print("Initializing slow buffer...should not see this at load from saved model!")
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)                    

                    #look ahead weight storage now in state dict 
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)

                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                                      

                #begin computations 
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                if amsgrad:
                        max_exp_avg_sq = state['max_exp_avg_sq']  
                                

                #compute variance mov avg
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                #compute mean moving avg
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                
               
                
                
                ##transformer
                if grad_transformer == 'square':
                    grad_tmp = grad**2
                elif grad_transformer == 'abs':
                    grad_tmp = grad.abs()


                exp_avg_sq.mul_(beta2).add_((1 - beta2)*grad_tmp)



                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denomc = max_exp_avg_sq.clone()
                else:
                    denomc = exp_avg_sq.clone()

                if grad_transformer == 'square':
                    #pdb.set_trace()
                    denomc.sqrt_()                 
                

                state['step'] += 1
               

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    
                    
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1                

                
                # ...let's use calibrated alr 
                if  group['transformer'] =='softplus':
                    sp = torch.nn.Softplus( smooth)
                    denomf = sp( denomc)
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denomf )  
                    
                else:
                    
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                

                p.data.copy_(p_data_fp32)

                #integrated look ahead...
                #we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    slow_p = state['slow_buffer'] #get access to slow param tensor
                    slow_p.add_(self.alpha, p.data - slow_p)  #(fast weights - slow weights) * alpha
                    p.data.copy_(slow_p)  #copy interpolated weights to RAdam param tensor

        return loss

class RangerQH(Optimizer):
    r"""Implements the QHAdam optimization algorithm `(Ma and Yarats, 2019)`_.
    Along with Hinton/Zhang Lookahead.
    Args:
        params (iterable):
            iterable of parameters to optimize or dicts defining parameter
            groups
        lr (float, optional): learning rate (:math:`\alpha` from the paper)
            (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of the gradient and its square
            (default: (0.9, 0.999))
        nus (Tuple[float, float], optional): immediate discount factors used to
            estimate the gradient and its square
            (default: (1.0, 1.0))
        eps (float, optional): term added to the denominator to improve
            numerical stability
            (default: 1e-8)
        weight_decay (float, optional): weight decay (default: 0.0)
        decouple_weight_decay (bool, optional): whether to decouple the weight
            decay from the gradient-based optimization step
            (default: False)
    Example:
        >>> optimizer = qhoptim.pyt.QHAdam(
        ...     model.parameters(),
        ...     lr=3e-4, nus=(0.8, 1.0), betas=(0.99, 0.999))
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    .. _`(Ma and Yarats, 2019)`: https://arxiv.org/abs/1810.06801
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        nus=(.7, 1.0),
        weight_decay=0.0,
        k=6,
        alpha=.5,
        decouple_weight_decay=False,
        eps=1e-8,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = {
            "lr": lr,
            "betas": betas,
            "nus": nus,
            "weight_decay": weight_decay,
            "decouple_weight_decay": decouple_weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)
        
        #look ahead params
        self.alpha = alpha
        self.k = k         
        

    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional):
                A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            nu1, nu2 = group["nus"]
            weight_decay = group["weight_decay"]
            decouple_weight_decay = group["decouple_weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                if d_p.is_sparse:
                    raise RuntimeError("QHAdam does not support sparse gradients")

                

                if weight_decay != 0:
                    if decouple_weight_decay:
                        p.data.mul_(1 - lr * weight_decay)
                    else:
                        d_p.add_(weight_decay, p.data)

                d_p_sq = d_p.mul(d_p)
                
                #prep for saved param loading
                param_state = self.state[p]

                if len(param_state) == 0:
                    param_state["beta1_weight"] = 0.0
                    param_state["beta2_weight"] = 0.0
                    param_state['step'] = 0
                    param_state["exp_avg"] = torch.zeros_like(p.data)
                    param_state["exp_avg_sq"] = torch.zeros_like(p.data)
                    #look ahead weight storage now in state dict 
                    param_state['slow_buffer'] = torch.empty_like(p.data)
                    param_state['slow_buffer'].copy_(p.data)
                
                
                param_state['step'] += 1                

                param_state["beta1_weight"] = 1.0 + beta1 * param_state["beta1_weight"]
                param_state["beta2_weight"] = 1.0 + beta2 * param_state["beta2_weight"]

                beta1_weight = param_state["beta1_weight"]
                beta2_weight = param_state["beta2_weight"]
                exp_avg = param_state["exp_avg"]
                exp_avg_sq = param_state["exp_avg_sq"]

                beta1_adj = 1.0 - (1.0 / beta1_weight)
                beta2_adj = 1.0 - (1.0 / beta2_weight)
                exp_avg.mul_(beta1_adj).add_(1.0 - beta1_adj, d_p)
                exp_avg_sq.mul_(beta2_adj).add_(1.0 - beta2_adj, d_p_sq)

                avg_grad = exp_avg.mul(nu1)
                if nu1 != 1.0:
                    avg_grad.add_(1.0 - nu1, d_p)

                avg_grad_rms = exp_avg_sq.mul(nu2)
                if nu2 != 1.0:
                    avg_grad_rms.add_(1.0 - nu2, d_p_sq)
                avg_grad_rms.sqrt_()
                if eps != 0.0:
                    avg_grad_rms.add_(eps)

                p.data.addcdiv_(-lr, avg_grad, avg_grad_rms)
                
                #integrated look ahead...
                #we do it at the param level instead of group level
                if param_state['step'] % self.k ==0: #group['k'] == 0:
                    slow_p = param_state['slow_buffer'] #get access to slow param tensor
                    slow_p.add_(self.alpha, p.data - slow_p)  #(fast weights - slow weights) * alpha
                    p.data.copy_(slow_p)  #copy interpolated weights to RAdam param tensor
                

        return loss

    @classmethod
    def _params_to_dict(cls, params):
        return {"lr": params.alpha, "nus": (params.nu1, params.nu2), "betas": (params.beta1, params.beta2)}