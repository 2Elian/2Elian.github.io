---
title: "从策略梯度PG到群组序列策略优化GSPO: 关于RLHF的一些思考"
summary: "LLM、RL、RLHF、PPO、GRPO、GSPO"
tags:
  - Large Language Model
date: 2025-08-29
---

本文将从LLM-RL的优化目标开始引入策略梯度，并逐渐过度到GSPO算法。同时会指出当前RLHF的问题以及自己做过的一些实验所总结的解决方法。

## 1. LLM中强化学习的优化目标

>  **符号约定**  
> $\tau = (s_0, a_0, s_1, a_1, \dots, s_t, a_t)$ 表示 LLM 采样得到的一条轨迹(序列);
>  
> 其中：  
> - $s_i$ 表示第 $i$ 步的状态，即当前已经生成的上下文;
> - $a_i$ 表示在状态 $s_i$ 下，模型选择的一个 token(即动作).
>  
> 例如，假设 prompt 是 **“请问 RLHF 是什么？”**：  
> - 初始状态 $s_0$ 就是这个 prompt 本身。  
> - LLM 根据 $s_0$ 计算下一个 token 的概率分布，假设 token **“R”** 的概率最高，并被选中，那么 $a_0 = \text{“R”}$。  
> - 于是得到的新状态 $s_1$ 就是：  
>   `"请问 RLHF 是什么？R"`  
>   R 就是 LLM 在第 0 步生成的第一个 token。  
>  
> 整个生成过程就是：  
> $s_i \;\;\xrightarrow{a_i}\;\; s_{i+1} \;\;\xrightarrow{a_{i+1}}\;\; s_{i+2} \;\;\dots$  
> 
> - $S$ 表示状态空间（所有可能的上下文）；  
> - $A$ 表示动作空间（通常就是整个词表 vocab）。  
>
> $\pi_\theta$ 表示需要优化的 LLM 策略模型（由参数 $\theta$ 决定）。  
>
> **Reward 与 Return**  
> - $r_t$ 表示在第 $t$ 步生成 token $a_t$ 后获得的即时奖励（reward），这个reward的目的是激励或者抑制当前token的输出概率，如果生成当前token获得的reward高，则代表鼓励当前token的生成，反之抑制当前token的生成。  
>   在 RLHF 中，这个 reward 通常由一个 **奖励模型** 或者 **人类反馈** 给出，例如对生成的完整回答打分，或者对某一步的 token 给出偏好信号。  
> - $R(\tau)$ 表示整条轨迹 $\tau$ 的回报（return）。常见的定义是所有 reward 的加权和：  
>   $$R(\tau) = \sum_{t=0}^{T} \gamma^t r_t$$
>   其中 $\gamma \in [0,1]$ 是折扣因子。  


> **优化目标**  
>
> $$\max_{\theta} \; \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$
>
> 含义如下：  
> - $\pi_\theta$ ：由参数 $\theta$ 控制的 LLM 策略（即模型本身）；  
> - $\tau \sim \pi_\theta$ ：轨迹 $\tau$ 是由策略 $\pi$ 采样得到的 (注意：我这里说的是$\pi$，而不是$\pi_{\theta}$。这是因为后续的PPO等算法，采样都是从老模型采样，这里暂时注意一下不要混淆！) 
> - $R(\tau)$ ：轨迹 $\tau$ 的回报（Return），衡量生成序列的整体质量；  
> - $\mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$ ：在策略 $\pi_\theta$ 下，生成轨迹的期望回报；  
> - $\max_\theta$ ：通过优化模型参数 $\theta$，让期望回报最大化。  
>
> 换句话说，RLHF 的目标就是：**找到一组模型参数，使得模型生成的序列在平均意义下尽可能获得更高的奖励**。
>

## 2. 策略梯度算法的引入和优缺点分析

我们的优化目标是：

$$\max_{\theta} \; \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

要想实现这一点，关键在于如何调整参数 $\theta$。  
一个自然的思路是：**对目标函数关于 $\theta$ 求梯度，然后沿着梯度上升的方向更新参数**。  
换句话说，只要我们能够写出：

$$\nabla_\theta \; \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$
就可以使用梯度上升来逐步优化策略。

这就是所谓的 **策略梯度 (Policy Gradient, PG)** 方法的基本出发点。

下面我们来求解这个梯度：

$$
\begin{aligned}
\nabla_\theta \; \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]
&= \nabla_\theta \sum_{\tau} \pi_\theta(\tau) R(\tau) 
&& \small\text{(期望定义)} \\
&= \sum_{\tau} \nabla_\theta \pi_\theta(\tau) R(\tau) 
&& \small\text{(把梯度移进去)} \\
&= \sum_{\tau} \pi_\theta(\tau)\, \nabla_\theta \log \pi_\theta(\tau) R(\tau) 
&& \small\text{(log-derivative trick)} \\
&= \mathbb{E}_{\tau \sim \pi_\theta}\!\left[ \nabla_\theta \log \pi_\theta(\tau)\, R(\tau) \right] 
&& \small\text{(转回期望)} \\
&= \frac{1}{N}\sum_{n=1}^{N}\!\left[ \nabla_\theta \log \pi_\theta(\tau^{n})\, R(\tau^{n}) \right] 
&& \small\text{(大数定律：经验平均)} \\
&= \frac{1}{N}\sum_{n=1}^{N}\!\left[ 
   \Big( \sum_{t=0}^{T_n} \nabla_\theta \log \pi_\theta(a_t^n \mid s_t^n) \Big)\, R(\tau^{n})
\right] 
&& \small\text{(轨迹 $\tau^n$ 的 token 展开)} 
\end{aligned}
$$

上面有一步我省略了：$\log \pi_\theta(\tau^n)$ 本质是各 token 概率的连乘，  
取对数后变成连加，因此可以写成 $\sum_{t=0}^{T_n} \log \pi_\theta(a_t^n \mid s_t^n)$。

总结一下，策略梯度可以写成：

$$\nabla_\theta \; \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] \;\approx\; \frac{1}{N}\sum_{n=1}^{N}\!\left[\Big( \sum_{t=0}^{T_n} \nabla_\theta \log \pi_\theta(a_t^n \mid s_t^n) \Big)\, R(\tau^{n})\right]$$

也就是说：根据上面的推导，我们只需要沿着该公式给出的 梯度上升方向 去更新参数 $\theta$，就能最大化期望回报。

但在神经网络的训练中，我们通常使用 梯度下降 这一优化范式。为了保持一致，可以将原本的最大化目标，转化为最小化其相反数。换句话说，就是把“梯度上升”问题改写为“梯度下降”问题来求解。

为了使用梯度下降，可以定义策略梯度的 loss 为：

$$\mathcal{L}(\theta) \;=\; - \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

对应的梯度就是：

$$\nabla_\theta \mathcal{L}(\theta) \;=\; - \nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] \;\approx\; - \frac{1}{N}\sum_{n=1}^{N}\!\left[\Big( \sum_{t=0}^{T_n} \nabla_\theta \log \pi_\theta(a_t^n \mid s_t^n) \Big)\, R(\tau^{n})\right]$$

这就是梯度策略的算法！

> **接下来我们看一下策略梯度的几个缺点**
>
> **1.当前的R($\tau$)是针对全局作用的，而非token粒度：**
>
> > 也就是说当一条轨迹$\tau$得到的return大于0，它会同时增加该轨迹中 所有状态下采取的动作 的概率；如果 return 小于 0，则会降低所有动作的概率。
> > 
> > 这种做法有一个关键的问题：
> > - 奖励分配不均匀：轨迹是由多个 token 构成的，但 return 只在整个序列上给出一个分数，导致无法精确区分到底是哪些 token 贡献了好的结果，哪些 token 其实是“拖后腿”的。
> > 
> > 那么如何解这个问题呢？
> > 
> > - 因为一个动作只能影响这个动作之后的reward而无法影响之前的，因此当前状态下采取一个动作的概率是增大还是减少，应该看采取了这个动作之后到序列结束所获得的reward累计值的大小。即$R(\tau) = \sum_{t'=t}^{T}r_{t'}$
> > 
> > - 当前状态下采取某个动作，应该只影响未来的几步，越到序列最后这种影响应该越小。即$R(\tau) = \sum_{t'=t}^{T}\gamma^{t'-t}r_{t'} = R_t$
>
> **2.Baseline问题：**
>
> > 在前面由问题1修正公式中，策略梯度估计是：$\frac{1}{N}\sum_{n=1}^{N}
\left[
   \sum_{t=0}^{T_n} \Big( \nabla_\theta \log \pi_\theta(a_t^n \mid s_t^n) \, R(\tau^n)\Big)
\right]$
> >
> > 这里直接用 $R(\tau^n)$ 来更新，会带来一个严重问题：方差过大。
因为 $R(\tau^n)$ 的数值可能在不同轨迹之间相差非常大，而 $\nabla_\theta \log \pi_\theta$ 又会放大这种差异，导致梯度估计波动剧烈，训练过程很不稳定。
> >
> > 解决方法：引入 Baseline。
我们可以在 return 外减去一个与状态有关的基准值 $b(s_t)$：$\frac{1}{N}\sum_{n=1}^{N}
\left[
   \sum_{t=0}^{T_n} \Big( \nabla_\theta \log \pi_\theta(a_t^n \mid s_t^n) \, (R_t^n - b(s_t^n))\Big)
\right]$
> >
> > 其中：
> > - $R_t^{n}$ 是从时刻 $t$ 开始的累计回报（discounted return）；
> > - $b(s_t^n)$ 是 baseline，可以理解为“对当前状态下能获得多少奖励的平均估计”。
> >
> > 直观解释：
> >
> > - 如果某个动作的回报 高于 baseline，说明它比平均水平更好 → 增大概率；
> > 
> > - 如果某个动作的回报 低于 baseline，说明它比平均水平更差 → 减小概率。
> >
> > 而这个baseline到底取什么，不同的算法有不同的方式。比如：PPO算法中取的就是优势函数

## 3. PPO算法

本质上，PPO 算法的核心就是在策略梯度中对 baseline 的设计改进。在 PPO 中，$(R_t^n - b(s_t^n))$ 被进一步推广为 优势函数 (Advantage Function)，用来衡量某个动作相对于平均水平的好坏。

而介绍优势函数之前，我们需要来了解一下RL的基础概念：

> **1.State Value Funcation 状态价值函数**
> >
> > 状态价值函数记作 $V(s_t)$，含义是：在状态 $s_t$ 下，按照当前策略 $\pi$ 行动，未来期望获得的 return：
> > $
V^\pi(s_t) = \mathbb{E}_\pi \big[ R_t \mid s_t \big], 
\quad R_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots
>>$
> >
> > 其中 $\gamma \in [0,1)$ 是折扣因子，用来让未来的奖励逐步衰减。
>
> **2. Action Value Function 动作价值函数**
>>
> >动作价值函数记作 $Q(s_t, a_t)$，含义是：在状态 $s_t$ 下采取动作 $a_t$，并且之后按照策略 $\pi$ 行动，未来期望获得的 return：
> > $$
Q^\pi(s_t, a_t) = \mathbb{E}_\pi \big[ R_t \mid s_t, a_t \big]
>> $$
>>
>> 用动作价值函数表达状态价值函数：
>>
>> $$
V^\pi(s_t) = \mathbb{E}_{a_t \sim \pi(\cdot \mid s_t)} \big[ Q^\pi(s_t, a_t) \big]$$
>>
>> 上面式子的含义是：状态价值 $V^\pi(s_t)$，等于在状态 $s_t$ 下，按照策略 $\pi$ 采样一个动作 $a_t$，然后在该动作下的动作价值 $Q^\pi(s_t,a_t)$ 的期望值。换句话说，状态的价值就是“在这个状态下，执行策略中可能动作的加权平均价值”，权重由策略 $\pi(a_t \mid s_t)$ 给出。
>>
>>$$
V^\pi(s_t) = \sum_{a_t} \pi(a_t \mid s_t) \, Q^\pi(s_t, a_t)$$
>
> **3. Advantage Function 优势函数**
> >
> > 优势函数记作 $A(s_t, a_t)$，它衡量在状态 $s_t$ 下，采取动作 $a_t$ 相比做其他动作好多少：
> >
> >$$ \begin{aligned}
A^\pi(s_t, a_t) &= Q^\pi(s_t, a_t) - V^\pi(s_t)
\end{aligned}$$
> >
> > 我们用这个优势函数作为策略梯度中$(R_t^n - b(s_t^n))$项的代替。如果 $A(s_t, a_t) > 0$，说明这个动作比平均水平更优，应该增大概率；如果 $A(s_t, a_t) < 0$，说明这个动作比平均水平更差，应该减小概率。

接下来，我们推导一下优势函数的具体表达：
> **$A^\pi(s_t, a_t)$的推导：**
> >$$ \begin{aligned}
A^\pi(s_t, a_t) &= Q^\pi(s_t, a_t) - V^\pi(s_t)
\end{aligned}$$
> >
> > $A^{\pi}_i$代表对状态价值的i次采样估计：
> >
> > $A^{\pi}_1 = r_t + \gamma V^{\pi}(s_{t+1}) - V^{\pi}(s_t)$
> >
> >$A^{\pi}_2 = r_t + \gamma (r_{t+1} + \gamma V^{\pi}(s_{t+2})) - V^{\pi}(s_t)$
>>
>> ...
>>
>> $A^{\pi}_T = r_t + \gamma r_{t+1} +  \gamma^2 r_{t+2} +  \gamma^3 r_{t+3} + ... +  \gamma^T r_{t+T} - V^{\pi}(s_t)$
>>
> 令$\delta_t = r_t + \gamma V^{\pi}(s_{t+1}) - V^{\pi}(s_t)$, 则$A^{\pi}_{1} = \delta_t$ , $A^{\pi}_{2} = \delta_t + \gamma \delta_{t+1}$, ... , $A^{\pi}_{T} = \delta_t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+2} + ...$

那么我们在实际应用的时候到底选几步采样来计算优势函数呢？PPO算法中说：我全都要！PPO中采用一个广义优势估计的方法，即GAE：

> $A^{\pi}(GAE) = (1-\lambda)(A^{\pi}_1 + \lambda A^{\pi}_2 + \lambda^2 A^{\pi}_3) + ...$, 其中超参数 $\lambda \in [0,1]$ 控制 bias-variance 权衡，实际训练中常取 $\lambda = 0.9$。
>
> 此时：PPO的雏形为：
> $$\frac{1}{N}\sum_{n=1}^{N}\left[\sum_{t=0}^{T_n} \Big( \nabla_\theta \log \pi_\theta(a_t^n \mid s_t^n) \, A^{GAE}_{\theta}(a_t^n , s_t^n)\Big)\right]$$

为什么说它是雏形呢？因为当前的目标函数还有一定的问题：

> 从上式可以看到，训练流程是 严格 on-policy 的：
先用当前策略采样一批数据 → 计算 log 概率和优势函数 → 得到 loss → 反向传播更新策略 → 下一轮再用新策略采样。这种方式的缺点是：训练效率低，数据利用率差。每个 batch 的数据只用来更新一次模型，更新完之后就丢弃了。
>
> 那么如何解决呢？最好的方式就是采样一次数据进行多次的policy model更新！
>
> 为了说清楚这个事情，我们先从一个概念出发：**重要性采样**
>>
>> 假设现在我有一个随机变量X服从于分布P(x), 然后我要计算f(x)的期望：$\mathbb{E}_{X\sim P(x)}[f(x)]$
>>
>> $\mathbb{E}_{X\sim P(x)}[f(x)] = \sum_{} f(x)p(x) = \sum_{} f(x)\frac{p(x)}{q(x)}q(x)$
>>
>>$= \mathbb{E}_{X\sim Q(x)}[f(x)\frac{p(x)}{q(x)}]$
>>
>> 即，我如果从P分布里面不好采样，我可以转向一个好采样的Q分布，然后计算f(x)与$\frac{p(x)}{q(x)}$的乘积的期望即可。
>>
>> 在 RLHF / PPO 中我们可以类比理解：原本只能从当前策略 $\pi_\theta$ 中采样（on-policy），导致每批数据只能用一次。现在通过重要性采样，我们可以允许从一个 参考分布（旧策略 $\pi_{\theta_{\text{old}}}$）中采样，然后再用 重要性权重 $\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}$ 来修正偏差。这样一来，一批数据可以反复用于多次更新策略模型，大幅提升样本效率。

通过上述分析，我们可以把PPO算法从On-policy的方式变成off-policy的方式：
$$
\begin{aligned}
&\frac{1}{N}\sum_{n=1}^{N}\!
\sum_{t=0}^{T_n}
\Big(
   \nabla_\theta \log \pi_\theta(a_t^n \mid s_t^n)
\Big)\;
\frac{\pi_\theta(a_t^n \mid s_t^n)}{\pi_{\theta_{\text{old}}}(a_t^n \mid s_t^n)}\;
A^{\text{GAE}}_{\theta_{\text{old}}}(s_t^n , a_t^n) \\[6pt]
&= \frac{1}{N}\sum_{n=1}^{N}\!
\sum_{t=0}^{T_n}
 \Bigg[
   \frac{\nabla_\theta \pi_\theta(a_t^n \mid s_t^n)}{\pi_\theta(a_t^n \mid s_t^n)}
   \frac{\pi_\theta(a_t^n \mid s_t^n)}{\pi_{\theta_{\text{old}}}(a_t^n \mid s_t^n)}
\Bigg]\;
A^{\text{GAE}}_{\theta_{\text{old}}}(s_t^n , a_t^n) \\[6pt]
&= \frac{1}{N}\sum_{n=1}^{N}\!
\sum_{t=0}^{T_n}
   \frac{\nabla_\theta \pi_\theta(a_t^n \mid s_t^n)}{\pi_{\theta_{\text{old}}}(a_t^n \mid s_t^n)}
A^{\text{GAE}}_{\theta_{\text{old}}}(s_t^n , a_t^n) \\[6pt]
\end{aligned}
$$

注意：上式中的 $\frac{\pi_\theta(a_t^n \mid s_t^n)}{\pi_{\theta_{\text{old}}}(a_t^n \mid s_t^n)}$ 就是一个 重要性采样的权重，用于修正采样分布与目标分布不一致所带来的偏差：
- 当这个比值 = 1 时，表示新旧策略在该状态下选择动作的概率相同，无需修正；
- 当这个比值 > 1 时，表示新策略更倾向于选择该动作，因此在梯度更新时会放大该动作的贡献；
- 当这个比值 < 1 时，表示新策略更少选择该动作，因此其贡献会被缩小。

现在，PPO算法已经完成了90%了，剩下的就是一些训练中的小修正：

> PPO 的目标里有比值：
>
> $$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} ,$$
>
> 它反映了 新策略和旧策略在同一个状态下选择相同行为的概率比。
>
> 问题是：如果 $r_t(\theta)$ 偏离 $1$ 太远（比如变得特别大或特别小），梯度更新就会出现 过大的方差 或者直接导致 策略崩溃（policy collapse）。
>
> 为了避免这种情况，PPO 引入了 clip 裁剪：
>
> $$
   \min\Big(
      r_t(\theta) A_t,\;\;
      \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t
   \Big)$$
>
>裁剪的直观意义是：
>- 当 $r_t(\theta)$ 在区间 $[1-\epsilon, 1+\epsilon]$ 内时，和原来一样；
> - 当 $r_t(\theta)$ 偏离过大时，就会被 钳住，防止更新过猛。
>
> 这样保证了策略更新的 “小步快走”：既能稳定，又能持续提升。



> 虽然有了 clip，但仍可能存在策略偏离旧策略过快的情况。为了进一步约束更新幅度，可以显式在 loss 里加一个正则化项：
>
> $$
   D_{\text{KL}}\!\big(
      \pi_{\theta_{\text{old}}}(\cdot \mid s_t)
      \;\|\;
      \pi_\theta(\cdot \mid s_t)
   \big)$$
>
> 这个方向的KL散度的含义是：让分布$\theta$尽可能的逼近分布$\theta_{old}$.
> - 当 KL 散度过大，说明新旧策略差异太大，训练可能不稳定,KL 项会把它拉回来，防止模型 “一步跨太大”；

最终PPO的损失函数就变成了：

$$
\begin{aligned}
L^{\text{PPO}}(\theta) 
= -\frac{1}{N}\sum_{n=1}^{N} \sum_{t=0}^{T_n} 
\Bigg[ 
   &\min\!\Bigg( 
      r_t^n(\theta) \, A^{\text{GAE}}_{\theta_{\text{old}}}(s_t^n , a_t^n), \\
   &\qquad\;\;
      \text{clip}\!\big(r_t^n(\theta), 1-\epsilon, 1+\epsilon \big) \, 
      A^{\text{GAE}}_{\theta_{\text{old}}}(s_t^n , a_t^n) 
   \Bigg) \\
   &\;-\; \beta \, D_{\text{KL}}\!\Big(
      \pi_{\theta_{\text{old}}}(\cdot \mid s_t^n)
      \;\|\;
      \pi_\theta(\cdot \mid s_t^n)
   \Big)
\Bigg]
\end{aligned}
$$

接下来，看一下这个复杂的PPO的训练流程：

```bash
初始化 πθ, πθ_old, Vφ from SFT之后的模型 ,初始化奖励模型来自于训练好的模型
for iteration = 1,2,... do
    采样轨迹 τ ~ πθ_old
    计算即时奖励 r_t
    计算状态价值 Vφ(s_t)
    计算优势函数 A_t (GAE)
    
    for epoch = 1..K do
        更新策略 θ
        更新价值网络 φ
    end
    
    更新参考模型 θ_old <- θ
end
```

## 4. KL散度的估计

在PPO中优势函数 $A_t$ 采用 **GAE（Generalized Advantage Estimation）** 递推公式计算：

$$
A_t = \delta_t + \gamma \lambda A_{t+1}
$$

该递推过程从时间步 $T$ 开始逆序计算(先算T，再算T-1，即逆序计算)，其中：

$$
A_T = \delta_T
$$

时间步 $t$ 的 TD 误差（Temporal Difference Error）计算公式为：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

Critic 的损失函数为：

$$
\text{critic\_loss} = \big(r_t + \gamma V(s_{t+1}) - V(s_t)\big)^2
$$

其中，值函数 $V(s_t)$ 由 Critic 模型估计。

奖励值 $r_t$ 计算如下：

$$
r(s_t, a_t) = \mathbb{I}(s_t = \text{[EOS]}) \, r(x, y) - \beta \, \text{KL}(t)
$$

其中 KL 散度计算为：

$$\text{KL}(t) = \log \frac{\pi^{\text{old}}_{\text{RL}}(a_t \mid s_t)}{\pi_{\text{SFT}}(a_t \mid s_t)}$$

> 说明：
> - 指示函数 $\mathbb{I}(s_t = \text{[EOS]})$ 表示仅当 $s_t$ 为句子结束标志（EOS）时取值为 1，否则为 0。这是因为一条序列实际上reward模型只会对最后一个token输出一个奖励值，token粒度的奖励值需要用kl散度来计算！  
> - 因此，在 PPO 中，KL 散度隐含在奖励函数的计算中。

在PPO算法中，KL散度项由于动作空间大小是vocab_size的，非常大。所以，真正计算的时候非常不好计算KL散度的值。为此，实际训练的时候会采用KL散度的估计值来代替KL散度。

> $KL(Q\mid P) = \sum q(x)log\frac{q(x)}{p(x)} = \mathbb{E}_{x\sim Q}[log\frac{q(x)}{p(x)}]$
> 
> **k1估计**
>> let $\frac{p(x)}{q(x)} = r$, k1 = -log(r)
>>
>> 则：$$\mathbb{E}_{x\sim Q}[k1] = \mathbb{E}_{x\sim Q}[-log(r)] = \mathbb{E}_{x\sim Q}[log\frac{q(x)}{p(x)}] = KL(Q\mid P)$$
>>
>>也就是说：构造的k1是KL散度的无偏估计（构造函数的期望值等于目标值）
>>
>>方差衡量的是一个随机变量的波动程度，也就是它的取值离期望值的平均偏离程度。公式是：
>>
>>$$
\mathrm{Var}[X] = \mathbb{E}[(X - \mathbb{E}[X])^2]$$
>>
>>- \(X\) 是一个随机变量
>>- 方差越大，说明 \(X\) 的取值在平均值周围波动得越厉害  
>>
>>在$
-\log r = -\log \frac{p(x)}{q(x)}$中 如果 \(q(x)\) 很小而 \(p(x)\) 相对大 → \(r\) 很大 → \(-log r\) 很小。如果 \(q(x)\) 很大而 \(p(x)\) 很小 → \(r\) 很小 → \(-log r\) 很大。这些极端值会导致单次采样的 \(-log r\) 离真正的期望（即 KL 值）很远，波动大 → 方差高。
>
> **k2估计**
> > k2 = $\frac{1}{2}(logr)^2$
>>
>>
>>$
\mathbb{E}_{x\sim Q}[k2] = \mathbb{E}_{x\sim Q}[\frac{1}{2}(logr)^2] = \frac{1}{2}\mathbb{E}_{x\sim Q}[(logr)^2]$
>>
>>$= \frac{1}{2}\sum q(x)(logr)^2  \neq KL(Q\mid P)$
>>
>> 因此k2估计器是KL散度的有偏估计
>>
>>但是因为$(logr)^2$将所有输入(正的和负的logr)都映射为正的输出。它避免了k1估计器中正负抵消的问题，并且不会产生极端大的负值，因此数值上更加稳定，波动更小。
>
> **k3估计**
> >
> >k3 = $r - 1 - logr$
> >
> >
> >$
> >\mathbb{E}_{x\sim Q}[k3] = \mathbb{E}_{x\sim Q}[r-1-logr] = \mathbb{E}_{x\sim Q}[r]  - \mathbb{E}_{x\sim Q}[1] - \mathbb{E}_{x\sim Q}[logr]$
> >
> >=$\sum q(x)*(\frac{p(x)}{q(x)}) - \sum q(x) -  \mathbb{E}_{x\sim Q}[logr]$
> >
> >=$1 - 1 - \mathbb{E}_{x\sim Q}[logr] = KL$
> >
> >因此k3也是一个无偏估计！
> >
> >但是函数$f(r) = r - 1 - logr$在r=1的时候取极小值0，即f(r)在定义域内横大于等于0，这意味着每一个样本对总估计的贡献都是非负的。也就是低方差的。兼具无偏和数值稳定的优良性质，通常是PPO实际训练时候的选择。
>

> **联系 between the k1, k2, and k3 Estimators**
> >
> > let $r = \frac{p(x)}{q(x)}$, and $\delta=r-1$.
> > 
> > 则$k1 = -log(\delta +1)$
> >
> >$k2 = \frac{1}{2}(log\frac{p(x)}{q(x)})^2 = \frac{1}{2}(log\frac{q(x)}{p(x)})^2 = \frac{1}{2}(log(\delta+1))^2$
> >
> >$k3 = \delta - log(\delta+1)$
>
> 接下来，发现k1, k2, k3中都有一个$log(\delta +1)$公共项，可以对它进行泰勒展开
>
> $log(\delta +1) = \delta-\frac{1}{2}\delta^2 + o(\delta^3)$
>
> 则$k1 \approx -\delta + \frac{1}{2}\delta^2$,
>
> $k2 \approx \frac{1}{2}\delta^2$,
>
> $k3 \approx \frac{1}{2}\delta^2$
> 
> 结论：
>
> $k_2$, $k_3$近似计算都可以由$-\log r = \log \frac{q(x)}{p(x)} \approx \delta - \frac{1}{2} \delta^2$的泰勒展开进行近似。其中，$k_1$ 和 $k_3$ 只使用一次泰勒展开，$k_2$ 则先对 $\log r$ 近似，再平方，等于两次近似。
$k_3$相当于在$k_1$的基础上加上了$\delta$，而：

$$
\mathbb{E}_{x \sim q}[\delta] 
= \mathbb{E}_{x \sim q}\!\left[\frac{p(x)}{q(x)} - 1\right] 
= \int p(x)\, dx - \int q(x)\, dx 
= 0.
$$

所以$k_3$相当于在$k_1$的基础上加上一个期望为$0$的变量，保证了$k_3 \geq 0$ (见上文的证明)，降低方差。





## 5. GRPO 

在PPO中, 训练的时候需要加载四个模型(policy model、state value model、reward model、ref model)，其中policy model、state value model的梯度要更新，reward model、ref model的梯度不需要进行更新。

在PPO训练的时候，这个statu value model的作用是参与优势函数的计算，因为里面有Q-V(动作价值函数-状态价值函数)，但这个计算出来的值是一个估计值，计算的就会有偏差。

于是deepseekAI lab就提出了GRPO，本质上是在优势函数上做文章，不需要估计state value function。

核心思想是：对于一个prompt，我采样一组response，用这一组的response的reward的相对值代表一个优势，即相对优势，相对这个概念是针对这一组的响应而言的。

首先将ppo算法写成如下的形式：

$$
J_{\mathrm{PPO}}(\theta) 
= \mathbb{E}_{q \sim P(Q), \; o \sim \pi_{\theta_{\mathrm{old}}}(O|q)}
\left[
  \alpha
\right]
$$

其中 $\alpha$ 定义为：
$$
\alpha = \frac{1}{|o|} \sum_{t=1}^{|o|} \min \left(
    r_t(\theta) \, A_t, \;
    \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \, A_t
\right)
$$

这里：$r_t(\theta) = \frac{\pi_\theta(o_t \mid q, o_{<t})}{\pi_{\theta_{\mathrm{old}}}(o_t \mid q, o_{<t})}$

> **关于clip和为什么取min的细节：**
>
> 之前看PPO的时候，以为PPO-Clip 引入了 clip 方法来控制策略（即动作概率）更新的幅度，确保新旧策略之间的变化在一定范围内，避免了过大的策略更新导致的性能下降或不稳定性。今天从另外一个角度来看看这个clip的真正寓意！
>
> PPO的核心clip机制：
>
> $$L_t^{\text{CLIP}} = \min \left( r_t(\theta) \cdot A_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t \right)$$
>
> 其中：$r_t(\theta) = \frac{\pi_\theta(o_t \mid q, o_{<t})}{\pi_{\theta_{\mathrm{old}}}(o_t \mid q, o_{<t})}$
>
>> - 当$A_{t}>0$的时候，说明执行当前动作$a_t$相较于其他动作要更好，因此要提升$\pi_{\theta}(a_t|s_t)$。但是也不能一味的提升它，这主要是由于以下两个原因
> > 1.  策略偏离旧策略太快 → 采样分布失效: 因为样本是从旧策略生成的，如果新策略变化太大就会导致新策略下的动作分布和旧策略差异很大，那么原来的优势估计就不再可靠。
> > 2.  梯度爆炸或不稳定: 比如某个动作at的概率原本很小=0.01, 如果无限制的提升其概率到0.9，这会导致梯度非常大，训练过程震荡甚至发散。
> > 3.  过度集中策略 → 探索不足: 如果每次都把概率无限提升，策略会迅速收敛到少数几个动作, 导致 探索不足，可能陷入局部最优
> >- 当$A_{t}<0$的时候，说明执行当前动作$a_t$相较于其他动作要更差，因此要降低$\pi_{\theta}(a_t|s_t)$。同样地，不能一味的降低它的概率，需要使用clip进行截断
>

> 对于初学者，这其中可能蕴含着两个疑惑：
> 1. 对概率比值做 clip，固定在阈值处，究竟意味着什么？
> 2. 如果 clip 是用于控制动作概率变化幅度的，那为什么还需要 min ？比如说按照下界进行 clip ，结果取完 min 操作保留的却还是未 clip 的值？
>
> > **问题1：clip操作的实际含义**
> > 
> > 如果执行了clip之后，将概率比 $r_t(\theta) = \frac{\pi_\theta(o_t \mid q, o_{<t})}{\pi_{\theta_{\mathrm{old}}}(o_t \mid q, o_{<t})}$ 固定在阈值之内变成常数，那么就意味着这个token不会参与梯度计算（参数没了，无法计算梯度），也就是说这个token不参与梯度更新，失效了！
>
> > **问题2：为什么需要min操作？**
> > 
> > 如果仅使用clip而不用min操作：
> > 
> > - **当$A_t > 0$时**：在禁止进一步增大高概率token概率的同时，小概率token的概率增长也被意外禁止了
> > - **当$A_t < 0$时**：类似问题
> > 
> > **具体例子**（$A_t > 0$情况）：当概率比率较小时（$r_t < 1-\varepsilon$），即当前策略输出该token的概率远小于旧策略时，clip操作会忽视该token不进行更新。但这是不合理的——我们应该对该小概率token进行更新，所以需要min操作使clip失效。$A_t < 0$情况同理。

GRPO定义如下：

$$
J_{\text{GRPO}}(\theta) = 
\mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O \mid q)}
\left[
\alpha
\right]
$$

其中 $\alpha$ 定义为：
$$
\begin{aligned}
\alpha &= \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} L_{i,t}(\theta) - \beta \, D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}}) \\[8pt]
L_{i,t}(\theta) &= \min \left(
  r_{i,t}(\theta) \, \hat{A}_{i,t}, \;
  \text{clip}(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon) \, \hat{A}_{i,t}
\right) \\[8pt]
r_{i,t}(\theta) &= \frac{\pi_\theta(o_{i,t} \mid q, o_{i,< t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,< t})}
\end{aligned}
$$

**GRPO中的优势函数**（相对优势，基于组内reward标准化）：
$$\hat{A}_{i,t} = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$$

## 6. GRPO会出现的一些问题

### 6.1 熵坍塌问题

   模型训练到后期的时候, 得到的rollout差别非常小, 也就是说：模型趋近于确定性. 这导致了计算优势的时候, 几乎为0, 导致模型无法更新：

   谁解决了这个问题呢？：DAPO

### 6.2 训练后期token裁剪率过高
   训练后期clip会把大多数token裁剪掉，导致参与梯度更新的token非常少，模型几乎不发生更新。另外，大量的token被裁剪的原因是：大多数token在某个节点脱离ref model太远！

   GSPO可解

   DCPO可解的更好

### 6.3 GRPO计算优势函数的时候, 组内采样的G需很大才行

   在群体规模不大的时候，其均值mean(r)的方差太大，导致模型训练不稳定。

   PVPO可解


## 7. 后续我的研究思路和成果

### 7.1 从熵正则角度来看GRPO


### 7.2 群组能量策略优化

   这是我的创新点