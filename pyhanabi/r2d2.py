# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from collections import OrderedDict
from ctypes import c_int16
from xmlrpc.client import Boolean
import torch
import torch.nn as nn
from typing import Tuple, Dict
from net import FFWDNet, PublicLSTMNet, LSTMNet, PPOPublicLSTMNet


class R2D2Agent(torch.jit.ScriptModule):
    __constants__ = [
        "vdn",
        "multi_step",
        "gamma",
        "eta",
        "boltzmann",
        "uniform_priority",
        "net",
    ]

    def __init__(
        self,
        ppo,
        perfect,
        clip_param,
        gae_lamda,
        c_1,
        c_2,
        train,
        vdn,
        multi_step,
        gamma,
        eta,
        device,
        in_dim,
        hid_dim,
        out_dim,
        net,
        num_lstm_layer,
        boltzmann_act,
        uniform_priority,
        off_belief,
        greedy=False,
        nhead=None,
        nlayer=None,
        max_len=None,
    ):
        super().__init__()
        if net == "ffwd":
            self.online_net = FFWDNet(in_dim, hid_dim, out_dim).to(device)
            self.target_net = FFWDNet(in_dim, hid_dim, out_dim).to(device)
        elif net == "publ-lstm" and ppo:
            self.online_net = PPOPublicLSTMNet(
                device, in_dim, hid_dim, out_dim, num_lstm_layer, perfect
            ).to(device)
            self.target_net = None
        elif net == "publ-lstm":
            self.online_net = PublicLSTMNet(
                device, in_dim, hid_dim, out_dim, num_lstm_layer
            ).to(device)
            self.target_net = PublicLSTMNet(
                device, in_dim, hid_dim, out_dim, num_lstm_layer
            ).to(device)
        elif net == "lstm":
            self.online_net = LSTMNet(
                device, in_dim, hid_dim, out_dim, num_lstm_layer
            ).to(device)
            self.target_net = LSTMNet(
                device, in_dim, hid_dim, out_dim, num_lstm_layer
            ).to(device)
        elif net == "transformer":
            self.online_net = TransformerNet(
                device, in_dim, hid_dim, out_dim, nhead, nlayer, max_len
            )
            self.target_net = TransformerNet(
                device, in_dim, hid_dim, out_dim, nhead, nlayer, max_len
            )
        else:
            assert False, f"{net} not implemented"

        if not ppo:
            for p in self.target_net.parameters():
                p.requires_grad = False

        self.ppo = ppo
        self.perfect = perfect
        self.clip_param = clip_param
        self.gae_lamda = gae_lamda
        self.c_1 = c_1
        self.c_2 = c_2
        self.train_ = train
        self.vdn = vdn
        self.multi_step = multi_step
        self.gamma = gamma
        self.eta = eta
        self.net = net
        self.num_lstm_layer = num_lstm_layer
        self.boltzmann = boltzmann_act
        self.uniform_priority = uniform_priority
        self.off_belief = off_belief
        self.greedy = greedy
        self.nhead = nhead
        self.nlayer = nlayer
        self.max_len = max_len

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        return self.online_net.get_h0(batchsize)

    def clone(self, device, overwrite=None):
        if overwrite is None:
            overwrite = {}
        cloned = type(self)(
            self.ppo,
            self.perfect,
            self.clip_param,
            self.gae_lamda,
            self.c_1,
            self.c_2,
            overwrite.get("train", self.train_),
            overwrite.get("vdn", self.vdn),
            self.multi_step,
            self.gamma,
            self.eta,
            device,
            self.online_net.in_dim,
            self.online_net.hid_dim,
            self.online_net.out_dim,
            self.net,
            self.num_lstm_layer,
            overwrite.get("boltzmann_act", self.boltzmann),
            self.uniform_priority,
            self.off_belief,
            self.greedy,
            nhead=self.nhead,
            nlayer=self.nlayer,
            max_len=self.max_len,
        )
        cloned.load_state_dict(self.state_dict())
        cloned.train(self.training)
        return cloned.to(device)

    def sync_target_with_online(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    @torch.jit.script_method
    def greedy_act(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # priv_s:[batch, dim]
        # publ_s:[batch, dim]
        # legal_move:[batch, out_dim]
        # hid:[batch, num_layer, num_player, dim]
        
        # adv:[batch, out_dim] new_hid:{'':[batch, num_layer, num_player, dim],}
        adv, new_hid = self.online_net.act_other(priv_s, publ_s, hid)
        # legal_adv:[batch, out_dim]
        legal_adv = (1 + adv - adv.min()) * legal_move
        # greedy_action:[batch]
        greedy_action = legal_adv.argmax(1).detach()
        return greedy_action, new_hid

    @torch.jit.script_method
    def boltzmann_act(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        temperature: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        temperature = temperature.unsqueeze(1)
        adv, new_hid = self.online_net.act_other(priv_s, publ_s, hid)
        assert adv.dim() == temperature.dim()
        logit = adv / temperature
        legal_logit = logit - (1 - legal_move) * 1e30
        assert legal_logit.dim() == 2
        prob = nn.functional.softmax(legal_logit, 1)
        action = prob.multinomial(1).squeeze(1).detach()
        return action, new_hid, prob

    @torch.jit.script_method
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Acts on the given obs, with eps-greedy policy.
        output: {'a' : actions}, a long Tensor of shape
            [batchsize] or [batchsize, num_player]
        """
        priv_s = obs["priv_s"]  # batch, dim
        publ_s = obs["publ_s"]  # batch, dim
        legal_move = obs["legal_move"]  # batch, out_dim
        if "eps" in obs:
            eps = obs["eps"].flatten(0, 1)  # batch
        else:
            eps = torch.zeros((priv_s.size(0),), device=priv_s.device)  # batch

        if self.vdn:
            bsize, num_player = obs["priv_s"].size()[:2]
            priv_s = obs["priv_s"].flatten(0, 1)
            publ_s = obs["publ_s"].flatten(0, 1)
            legal_move = obs["legal_move"].flatten(0, 1)
        else:
            bsize, num_player = obs["priv_s"].size()[0], 1

        #  [batch, num_layer, num_player, dim]
        hid = {"h0": obs["h0"], "c0": obs["c0"]}

        if self.boltzmann:
            temp = obs["temperature"].flatten(0, 1)
            greedy_action, new_hid, prob = self.boltzmann_act(priv_s, publ_s, legal_move, temp, hid)
            reply = {"prob": prob}
            old_probs = torch.zeros_like(legal_move)
            old_values = torch.zeros_like(greedy_action)
        else:
            if self.ppo:
                if self.perfect and self.train_:
                    # print(obs["priv_s"].shape)
                    # print(obs["own_hand"].shape)
                    if obs["own_hand"].dim() == 2:
                        own_hand = obs["own_hand"]
                    elif obs["own_hand"].dim() == 4:
                        own_hand = obs["own_hand"].flatten(2, 3).contiguous()
                    else:
                        own_hand = obs["own_hand"]
                    # print("own_hand:", own_hand)
                    perf_s = torch.cat([obs["priv_s"], own_hand], dim=-1)
                    # print(f"{perf_s.shape[-1]} {self.train_} {own_hand.shape}")
                    assert perf_s.shape[-1] == 783
                else:
                    perf_s = torch.zeros_like(priv_s)
                # greedy_action:[batch]
                # new_hid:{'':[batch, num_layer, num_player, dim],}
                # old_probs:[batch, out_dim]
                # old_values:[batch]
                greedy_action, new_hid, old_probs, old_values = self.online_net.act(priv_s, publ_s, hid, legal_move, perf_s, self.train_)
            else:
                # greedy_action: [batch]
                # new_hid:{'':[batch, num_layer, num_player, dim],}
                greedy_action, new_hid = self.greedy_act(priv_s, publ_s, legal_move, hid)
                old_probs = torch.zeros_like(legal_move)
                old_values = torch.zeros_like(greedy_action)
            reply = {}

        if self.greedy:
            action = greedy_action
            random_action = torch.zeros_like(greedy_action)
            rand = torch.zeros_like(greedy_action)
        else:
            # random_action:[batch] sample by chance(actually uniform dist)
            random_action = legal_move.multinomial(1).squeeze(1)
            # rand:[batch]
            rand = torch.rand(greedy_action.size(), device=greedy_action.device)
            assert rand.size() == eps.size()
            rand = (rand < eps).float()  # rate "eps" to explore
            # action:[batch]
            action = (greedy_action * (1 - rand) + random_action * rand).detach().long()

        if self.vdn:
            action = action.view(bsize, num_player)
            greedy_action = greedy_action.view(bsize, num_player)
            # rand = rand.view(bsize, num_player)
        
        check = legal_move.detach().clone().gather(1, action.unsqueeze(1)).squeeze(1)
        a = torch.sum(check, dtype=torch.float)
        b = torch.tensor(bsize, device=a.device, dtype=torch.float)
        if a - b != 0:
            print("old_probs:", old_probs)
            print("check:", check)
            print("legal_move:", legal_move)
            print("rand:", rand)
            print("greedy_action:", greedy_action)
            print("random_action:", random_action)
            print("action:", action)
        assert a - b == 0
        
        # batch
        reply["a"] = action.detach().cpu()
        # batch, num_layer, num_player, dim
        reply["h0"] = new_hid["h0"].detach().cpu()
        # batch, num_layer, num_player, dim
        reply["c0"] = new_hid["c0"].detach().cpu()
        reply["old_probs"] = old_probs.cpu()
        if self.train_:
            reply["old_values"] = old_values.cpu()
        return reply

    @torch.jit.script_method
    def compute_target(
        self, input_: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """compute the 1-step TD-target for each sample at each time in the batch
        # *used for blueprint policy's reward computation as in RLSearch
        
        Args:
            input_ (Dict[str, torch.Tensor]): _description_

        Returns:
            Dict[str, torch.Tensor]: _description_
        """
        assert self.multi_step == 1
        priv_s = input_["priv_s"]
        publ_s = input_["publ_s"]
        legal_move = input_["legal_move"]
        act_hid = {  # hid:[batch, num_layer, num_player, dim]
            "h0": input_["h0"],
            "c0": input_["c0"],
        }
        fwd_hid = {  # hid:[num_layer, batch*num_player, dim]
            "h0": input_["h0"].transpose(0, 1).flatten(1, 2).contiguous(),
            "c0": input_["c0"].transpose(0, 1).flatten(1, 2).contiguous(),
        }
        reward = input_["reward"]
        terminal = input_["terminal"]

        if self.boltzmann:
            temp = input_["temperature"].flatten(0, 1)
            next_a, _, next_pa = self.boltzmann_act(
                priv_s, publ_s, legal_move, temp, act_hid
            )
            # !This is wrong
            next_q = self.online_net.forward_other(priv_s, publ_s, legal_move, next_a, fwd_hid)[2]
            qa = (next_q * next_pa).sum(1)
        else:
            if self.ppo:
                pass
                qa = torch.zeros_like(reward)
            else:
                # next_a [batch]
                next_a = self.greedy_act(priv_s, publ_s, legal_move, act_hid)[0]
                # !This is wrong
                # qa: [batch]
                qa = self.online_net.forward_other(priv_s, publ_s, legal_move, next_a, fwd_hid)[0]

        assert reward.size() == qa.size()
        
        if self.ppo:
            pass

        # 1-step TD-target: [batch]
        target = reward + (1 - terminal) * self.gamma * qa
        return {"target": target.detach()}

    @torch.jit.script_method
    def compute_priority(
        self, input_: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """compute the priority for each sample in the batch
        # *used for new samples' priority as in replay_buffer

        Args:
            input_ (Dict[str, torch.Tensor]): _description_

        Returns:
            Dict[str, torch.Tensor]: _description_
        """
        if self.uniform_priority:
            return {"priority": torch.ones_like(input_["reward"].sum(1))}  # batch

        # swap batch_dim and seq_dim
        # [batch, seq_len] -> [seq_len, batch]
        for k, v in input_.items():
            if k != "seq_len":
                input_[k] = v.transpose(0, 1).contiguous()

        obs = {
            "priv_s": input_["priv_s"],
            "publ_s": input_["publ_s"],
            "legal_move": input_["legal_move"],
            "own_hand": input_["own_hand"],
        }
        if self.boltzmann:
            obs["temperature"] = input_["temperature"]

        if self.off_belief:
            obs["target"] = input_["target"]

        hid = {"h0": input_["h0"], "c0": input_["c0"]}
        action = {"a": input_["a"], "old_probs": input_["old_probs"], "old_values": input_["old_values"]}
        reward = input_["reward"]
        terminal = input_["terminal"]
        bootstrap = input_["bootstrap"]
        seq_len = input_["seq_len"]
        # err: [seq_len, batch]
        if self.ppo:
            _, _, err, _ = self.td_error(
                obs, hid, action, reward, terminal, bootstrap, seq_len
            )
        else:
            err, _, _, _ = self.td_error(
                obs, hid, action, reward, terminal, bootstrap, seq_len
            )
        priority = err.abs()
        priority = self.aggregate_priority(priority, seq_len).detach().cpu()
        return {"priority": priority}

    @torch.jit.script_method
    def td_error(
        self,
        obs: Dict[str, torch.Tensor],
        hid: Dict[str, torch.Tensor],
        action: Dict[str, torch.Tensor],
        reward: torch.Tensor,
        terminal: torch.Tensor,
        bootstrap: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        # print(action["a"].shape)
        # print(obs["priv_s"].shape)
        max_seq_len = obs["priv_s"].size(0)
        priv_s = obs["priv_s"]
        publ_s = obs["publ_s"]
        legal_move = obs["legal_move"]
        action_ = action["a"]
        old_probs = action["old_probs"]
        old_values = action["old_values"]

        # hid size: [num_layer, batch, num_player, dim]
        # -> [num_layer, batch*num_player, dim]
        for k, v in hid.items():
            hid[k] = v.flatten(1, 2).contiguous()

        bsize, num_player = priv_s.size(1), 1
        if self.vdn:
            num_player = priv_s.size(2)
            priv_s = priv_s.flatten(1, 2)
            publ_s = publ_s.flatten(1, 2)
            legal_move = legal_move.flatten(1, 2)
            action_ = action_.flatten(1, 2)

        # this only works because the trajectories are padded,
        # i.e. no terminal in the middle
        # !in the next few lines, "seq_len" dim means "max_seq_len" in the former context
        
        if self.ppo:
            if self.perfect and self.train_:
                if obs["own_hand"].dim() == 2:
                    own_hand = obs["own_hand"]
                elif obs["own_hand"].dim() == 4:
                    own_hand = obs["own_hand"].flatten(2, 3).contiguous()
                else:
                    own_hand = obs["own_hand"]
                perf_s = torch.cat([obs["priv_s"], own_hand], dim=-1)
                assert perf_s.shape[-1] == 783
            else:
                perf_s = torch.zeros_like(priv_s)
            # values: [seq_len, batch]
            # greedy_a: [seq_len, batch]
            # probs: [seq_len, batch, num_action]
            # lstm_o: [seq_len, batch, dim]
            values, greedy_a, probs, lstm_o = self.online_net(
                priv_s, publ_s, legal_move, action_, hid, perf_s, self.train_
            )
            online_qa = torch.zeros_like(greedy_a)
            online_q = torch.zeros_like(probs)
        else:
            # online_qa: [seq_len, batch]
            # greedy_a: [seq_len, batch]
            # online_q: [seq_len, batch, num_action]
            # lstm_o: [seq_len, batch, dim]
            online_qa, greedy_a, online_q, lstm_o = self.online_net.forward_other(
                priv_s, publ_s, legal_move, action_, hid
            )
            values = torch.zeros_like(greedy_a)
            probs = torch.zeros_like(online_q)

        loss_dict = dict()
        if self.off_belief:
            target = obs["target"]
            value_target = torch.zeros_like(target)
            old_values = torch.zeros_like(target)
            loss = torch.zeros_like(target)
            loss_p = torch.zeros_like(target)
            loss_v = torch.zeros_like(target)
            loss_entropy = torch.zeros_like(target)
            err = torch.zeros_like(target)
            advs = torch.zeros_like(target)
            ratios = torch.zeros_like(target)
            clipped_ratios = torch.zeros_like(target)
        elif self.ppo:
            # Calculate GAE Advantage
            rewards = reward.detach().clone()  # [seq_len, batch]
            gae = torch.zeros_like(rewards)  # [seq_len, batch]
            nextgae = torch.zeros_like(rewards)  # [seq_len, batch]
            returns = torch.zeros_like(rewards)
            nextreturns = torch.zeros_like(rewards)
            # print("rewards.shape", rewards.shape)
            # print("nextgae.shape", nextgae.shape)
            
            nextvalues = torch.zeros_like(old_values)
            nextvalues[:-self.multi_step, :] = old_values[self.multi_step:, :]
            # print("rewards:\n", rewards)
            # print("old_values:\n", old_values)
            # print("nextvalues:\n", nextvalues)
            # print("bootstrap:\n", bootstrap)
            # print("seq_len:\n", seq_len)
            
            value_target = rewards + bootstrap * (self.gamma ** self.multi_step) * nextvalues
            delta = value_target - old_values
            # print("value_target:\n", value_target)
            # print("delta:\n", delta)
            # print("rate:", (self.gamma ** self.multi_step) * self.gae_lamda)
            for _ in torch.arange(0, old_values.shape[0], device=seq_len.device):
                nextgae[:-self.multi_step, :] = gae[self.multi_step:, :]
                gae = delta + bootstrap * (self.gamma ** self.multi_step) * self.gae_lamda * nextgae
                nextreturns[:-self.multi_step, :] = returns[self.multi_step:, :]
                returns = rewards + bootstrap * (self.gamma ** self.multi_step) * nextreturns
            advs = gae
            advs = (advs - torch.mean(advs, dim=-1, keepdim=True)) / (torch.std(advs, dim=-1, keepdim=True) + 1e-6)
            
            # gae = delta.detach().clone()
            # returns = rewards.detach().clone()
            # for i in torch.arange(0, old_values.shape[1], device=seq_len.device):
            #     for t in torch.arange(old_values.shape[0] - 2, -1, -1, device=seq_len.device):
            #         if bootstrap[t, i] == 0:
            #             continue
            #         else:
            #             gae[t, i] = gae[t, i] + bootstrap[t, i] * (self.gamma ** self.multi_step) * self.gae_lamda * gae[t + 1, i]
            #             returns[t, i] = returns[t, i] + bootstrap[t, i] * (self.gamma ** self.multi_step) * returns[t + 1, i]
            # advs = gae
            # print("advs1:\n", advs)
            # print("advs2:\n", advs2)
            
            # policy loss
            # [seq_len, batch]
            probs_a = probs.gather(2, action_.unsqueeze(2)).squeeze(2)
            old_probs_a = old_probs.gather(2, action_.unsqueeze(2)).squeeze(2)
            ratios = torch.exp(torch.log(torch.clamp(probs_a, 1e-10, 1.0)) - torch.log(torch.clamp(old_probs_a, 1e-10, 1.0)))
            loss_p1 = advs.detach() * torch.clamp(ratios, 0., 3.)
            loss_p1 = torch.where(ratios==3., 0., loss_p1)
            clipped_ratios = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param)
            loss_p2 = torch.multiply(advs.detach(), clipped_ratios)
            loss_p = torch.minimum(loss_p1, loss_p2)
            
            # value loss
            # [seq_len, batch]
            value_pred_clipped = old_values + (values - old_values).clamp(-self.clip_param, self.clip_param)
            err1 = values - returns
            value_losses = nn.functional.smooth_l1_loss(err1, torch.zeros_like(err1), reduction="none")
            err2 = value_pred_clipped - returns
            value_losses_clipped = nn.functional.smooth_l1_loss(err2, torch.zeros_like(err2), reduction="none")
            loss_v = torch.maximum(value_losses, value_losses_clipped)
            err = torch.maximum(err1.abs(), err2.abs())
            
            # entropy loss
            # [seq_len, batch]
            loss_entropy = -torch.sum(probs * torch.log(torch.clamp(probs, 1e-10, 1.0)), dim=-1)
            
            # total loss
            # [seq_len, batch]
            loss = -loss_p + self.c_1 * loss_v - self.c_2 * loss_entropy
            
            target = torch.zeros_like(loss)
            online_qa = torch.zeros_like(loss)
        else:
            # !This is wrong
            target_qa, _, target_q, _ = self.online_net.forward_other(
                priv_s, publ_s, legal_move, greedy_a, hid
            )

            if self.boltzmann:
                temperature = obs["temperature"].flatten(1, 2).unsqueeze(2)
                # online_q: [seq_len, bathc * num_player, num_action]
                logit = online_q / temperature.clamp(min=1e-6)
                # logit: [seq_len, batch * num_player, num_action]
                legal_logit = logit - (1 - legal_move) * 1e30
                assert legal_logit.dim() == 3
                pa = nn.functional.softmax(legal_logit, 2).detach()
                # pa: [seq_len, batch * num_player, num_action]

                assert target_q.size() == pa.size()
                target_qa = (pa * target_q).sum(-1).detach()
                assert online_qa.size() == target_qa.size()

            if self.vdn:
                online_qa = online_qa.view(max_seq_len, bsize, num_player).sum(-1)
                target_qa = target_qa.view(max_seq_len, bsize, num_player).sum(-1)
                lstm_o = lstm_o.view(max_seq_len, bsize, num_player, -1)

            target_qa = torch.cat(
                [target_qa[self.multi_step :], target_qa[: self.multi_step]], 0
            )
            target_qa[-self.multi_step :] = 0
            assert target_qa.size() == reward.size()
            # This reward is sum discounted reward form t_current to t_{multi_step - 1}
            target = reward + bootstrap * (self.gamma ** self.multi_step) * target_qa
            loss = torch.zeros_like(target)
            value_target = torch.zeros_like(target)
            old_values = torch.zeros_like(target)
            loss_p = torch.zeros_like(target)
            loss_v = torch.zeros_like(target)
            loss_entropy = torch.zeros_like(target)
            err = torch.zeros_like(target)
            advs = torch.zeros_like(target)
            ratios = torch.zeros_like(target)
            clipped_ratios = torch.zeros_like(target)
        # seq_len: [batch]
        mask = torch.arange(0, max_seq_len, device=seq_len.device)
        # mask: [seq_len, 1]  seq_len: [1, batch]
        mask = (mask.unsqueeze(1) < seq_len.unsqueeze(0)).float()
        
        loss_dict["p"] = -loss_p.detach() * mask
        loss_dict["v"] = loss_v.detach() * mask
        loss_dict["e"] = -loss_entropy.detach() * mask
        loss_dict["adv"] = advs.detach() * mask
        loss_dict["ratios"] = ratios.detach() * mask
        loss_dict["ratios_clip"] = clipped_ratios.detach() * mask
        # e.g. mask [5, 1] seq_len: [1, 3]
        # mask [[0], [1], [2], [3], [4]]
        # seq_len [[2, 3, 1]]
        # mask [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]
        # seq_len [[2, 3, 1], [2, 3, 1], [2, 3, 1], [2, 3, 1], [2, 3, 1]]
        # mask  [[1, 1, 1], [1, 1, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]]
        # This is a process that mask the valid seq_len for each data of the batch.
        # mask: [seq_len, batch]
        # online_qa: [seq_len, batch]
        # target: [seq_len, batch]
        # e.g. if a data seq_len is 5, it only has loss for the first 5 steps, 
        # and 0 loss for the rest of steps(some number like 75).
        if self.ppo:
            # [seq_len, batch]
            loss = loss * mask
            err = err * mask
            return loss, loss_dict, err, lstm_o
        else:
            err = (target.detach() - online_qa) * mask
        if self.off_belief and "valid_fict" in obs:
            err = err * obs["valid_fict"]
        
        # err: [seq_len, batch]
        # lstm_o: [seq_len, batch, dim]
        # online_q: [seq_len, batch, num_action]
        return err, loss_dict, lstm_o, online_q

    def aux_task_iql(self, lstm_o, hand, seq_len, rl_loss_size, stat):
        # lstm_o: [seq_len, batch, dim]
        # hand: [seq_len, batch, 5, 3]
        # seq_len: [batch]
        # rl_loss: [batch]
        seq_size, bsize, _ = hand.size()
        own_hand = hand.view(seq_size, bsize, 5, 3)
        # # own_hand_slot_mask: [seq_len, batch, 5]
        own_hand_slot_mask = own_hand.sum(3)
        pred_loss1, avg_xent1, _, _ = self.online_net.pred_loss_1st(
            lstm_o, own_hand, own_hand_slot_mask, seq_len
        )
        assert pred_loss1.size() == rl_loss_size
        stat["aux"].feed(avg_xent1)
        return pred_loss1

    def aux_task_vdn(self, lstm_o, hand, t, seq_len, rl_loss_size, stat):
        """1st and 2nd order aux task used in VDN"""
        seq_size, bsize, num_player, _ = hand.size()
        own_hand = hand.view(seq_size, bsize, num_player, 5, 3)
        own_hand_slot_mask = own_hand.sum(4)
        pred_loss1, avg_xent1, belief1, _ = self.online_net.pred_loss_1st(
            lstm_o, own_hand, own_hand_slot_mask, seq_len
        )
        assert pred_loss1.size() == rl_loss_size

        stat["aux"].feed(avg_xent1)
        return pred_loss1

    def aggregate_priority(self, priority, seq_len):
        # priority: [seq_len, batch]
        # p_mean: [batch]
        p_mean = priority.sum(0) / seq_len
        # .max -> (values, indices)
        p_max = priority.max(0)[0]
        # agg_priority: [batch]
        agg_priority = self.eta * p_max + (1.0 - self.eta) * p_mean
        return agg_priority

    def loss(self, batch, aux_weight, stat):
        if not self.ppo:
            # err: [seq_len, batch]
            # lstm_o: [seq_len, batch, dim]
            # online_q: [seq_len, batch, num_action]
            err, loss_dict, lstm_o, online_q = self.td_error(
                batch.obs,
                batch.h0,
                batch.action,
                batch.reward,
                batch.terminal,
                batch.bootstrap,
                batch.seq_len,
            )
            # smooth_l1_loss: L(err) -> 0
            rl_loss = nn.functional.smooth_l1_loss(
                err, torch.zeros_like(err), reduction="none"
            )
        else:
            rl_loss, loss_dict, err, lstm_o = self.td_error(
                batch.obs,
                batch.h0,
                batch.action,
                batch.reward,
                batch.terminal,
                batch.bootstrap,
                batch.seq_len,
            )
        # rl_loss: [batch]
        rl_loss = rl_loss.sum(0)
        stat["rl_loss"].feed((rl_loss / batch.seq_len).mean().item())  # rl_loss: double
        stat["loss_p"].feed((loss_dict["p"].sum(0) / batch.seq_len).mean().item())  # loss_p: double
        stat["loss_v"].feed((loss_dict["v"].sum(0) / batch.seq_len).mean().item())  # loss_v: double
        stat["loss_e"].feed((loss_dict["e"].sum(0) / batch.seq_len).mean().item())  # loss_e: double
        stat["adv"].feed((loss_dict["adv"].sum(0) / batch.seq_len).mean().item())  # adv: double
        stat["ratios"].feed((loss_dict["ratios"].sum(0) / batch.seq_len).mean().item())  # ratios: double
        stat["ratios_clip"].feed((loss_dict["ratios_clip"].sum(0) / batch.seq_len).mean().item())  # ratios_clip: double
        # priority: [seq_len, batch]
        priority = err.abs()
        # priority: [batch]
        priority = self.aggregate_priority(priority, batch.seq_len).detach().cpu()
        stat["priority"].feed(priority.mean().item())  # priority: double
        
        loss = rl_loss
        if aux_weight <= 0:
            if self.ppo:
                return loss, priority
            # rl_loss: [batch]  priority: [batch]  online_q: [seq_len, batch, num_action]
            return loss, priority, online_q

        if self.vdn:
            pred1 = self.aux_task_vdn(
                lstm_o,
                batch.obs["own_hand"],
                batch.obs["temperature"],
                batch.seq_len,
                rl_loss.size(),
                stat,
            )
            loss = rl_loss + aux_weight * pred1
        else:
            pred = self.aux_task_iql(
                lstm_o,  # lstm_o: [seq_len, batch, dim]
                batch.obs["own_hand"],  # own_hand: [seq_len, batch, 5, 3]
                batch.seq_len,  # seq_len: [batch]
                rl_loss.size(),  # rl_loss: [batch]
                stat,
            )
            loss = rl_loss + aux_weight * pred
        
        return loss, priority, online_q

    def behavior_clone_loss(self, online_q, batch, t, clone_bot, stat):
        max_seq_len = batch.obs["priv_s"].size(0)
        priv_s = batch.obs["priv_s"]
        publ_s = batch.obs["publ_s"]
        legal_move = batch.obs["legal_move"]

        bsize, num_player = priv_s.size(1), 1
        if self.vdn:
            num_player = priv_s.size(2)
            priv_s = priv_s.flatten(1, 2)
            publ_s = publ_s.flatten(1, 2)
            legal_move = legal_move.flatten(1, 2)

        with torch.no_grad():
            target_logit, _ = clone_bot(priv_s, publ_s, None)
            target_logit = target_logit - (1 - legal_move) * 1e10
            target = nn.functional.softmax(target_logit, 2)

        logit = online_q / t
        # logit: [seq_len, batch * num_player, num_action]
        legal_logit = logit - (1 - legal_move) * 1e10
        log_distq = nn.functional.log_softmax(legal_logit, 2)

        assert log_distq.size() == target.size()
        assert log_distq.size() == legal_move.size()
        xent = -(target.detach() * log_distq).sum(2) / legal_move.sum(2).clamp(min=1e-3)
        if self.vdn:
            xent = xent.view(max_seq_len, bsize, num_player).sum(2)

        mask = torch.arange(0, max_seq_len, device=batch.seq_len.device)
        mask = (mask.unsqueeze(1) < batch.seq_len.unsqueeze(0)).float()
        assert xent.size() == mask.size()
        xent = xent * mask
        xent = xent.sum(0)
        stat["bc_loss"].feed(xent.mean().detach())
        return xent
