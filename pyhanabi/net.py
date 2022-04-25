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
from xmlrpc.client import Boolean
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
import common_utils
import math


@torch.jit.script
def duel(v: torch.Tensor, a: torch.Tensor, legal_move: torch.Tensor) -> torch.Tensor:
    assert a.size() == legal_move.size()
    assert legal_move.dim() == 3  # seq, batch, dim
    legal_a = a * legal_move
    q = v + legal_a - legal_a.mean(2, keepdim=True)
    return q


def cross_entropy(net, lstm_o, target_p, hand_slot_mask, seq_len):
    # net: nn.Linear(dim, 5 * 3)
    # lstm_o: [seq_len, batch, dim]
    # target_p: [seq_len, batch, (num_player,) 5, 3]
    # hand_slot_mask: [seq_len, batch, (num_player,) 5]
    # seq_len: [batch]
    
    # logit: [seq_len, batch, 5, 3]
    logit = net(lstm_o).view(target_p.size())
    # q: [seq_len, batch, 5, 3]
    q = nn.functional.softmax(logit, -1)
    logq = nn.functional.log_softmax(logit, -1)
    # [seq_len, batch, 5]
    plogq = (target_p * logq).sum(-1)
    # hand_slot_mask: if the target in none, then there is no loss
    # [seq_len, batch]
    xent = -(plogq * hand_slot_mask).sum(-1) / hand_slot_mask.sum(-1).clamp(min=1e-6)

    if xent.dim() == 3:
        # [seq, batch, num_player]
        xent = xent.mean(2)

    # save before sum out
    seq_xent = xent  # [seq_len, batch]
    xent = xent.sum(0)  # [batch]
    assert xent.size() == seq_len.size()
    avg_xent = (xent / seq_len).mean().item()  # [batch]
    return xent, avg_xent, q, seq_xent.detach()


class FFWDNet(torch.jit.ScriptModule):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        # for backward compatibility
        if isinstance(in_dim, int):
            assert in_dim == 783
            self.in_dim = in_dim
            self.priv_in_dim = in_dim - 125
            self.publ_in_dim = in_dim - 2 * 125
        else:
            self.in_dim = in_dim
            self.priv_in_dim = in_dim[1]
            self.publ_in_dim = in_dim[2]

        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.net = nn.Sequential(
            nn.Linear(self.priv_in_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
        )

        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.out_dim)
        # for aux task
        self.pred_1st = nn.Linear(self.hid_dim, 5 * 3)

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        """fake, only for compatibility"""
        shape = (1, batchsize, 1)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def act(
        self, priv_s: torch.Tensor, publ_s: torch.Tensor, hid: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert priv_s.dim() == 2, "dim should be 2, [batch, dim], get %d" % priv_s.dim()
        o = self.net(priv_s)
        a = self.fc_a(o)
        return a, hid

    @torch.jit.script_method
    def forward(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        action: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
            priv_s.dim() == 3 or priv_s.dim() == 2
        ), "dim = 3/2, [seq_len(optional), batch, dim]"

        o = self.net(priv_s)
        a = self.fc_a(o)
        v = self.fc_v(o)
        q = duel(v, a, legal_move)

        # q: [(seq_len), batch, num_action]
        # action: [seq_len, batch]
        qa = q.gather(-1, action.unsqueeze(-1)).squeeze(-1)

        assert q.size() == legal_move.size()
        legal_q = (1 + q - q.min()) * legal_move
        # greedy_action: [(seq_len), batch]
        greedy_action = legal_q.argmax(-1).detach()
        return qa, greedy_action, q, o

    def pred_loss_1st(self, o, target, hand_slot_mask, seq_len):
        return cross_entropy(self.pred_1st, o, target, hand_slot_mask, seq_len)


class LSTMNet(torch.jit.ScriptModule):
    __constants__ = ["hid_dim", "out_dim", "num_lstm_layer"]

    def __init__(self, device, in_dim, hid_dim, out_dim, num_lstm_layer):
        super().__init__()
        # for backward compatibility
        if isinstance(in_dim, int):
            assert in_dim == 783
            self.in_dim = in_dim
            self.priv_in_dim = in_dim - 125
            self.publ_in_dim = in_dim - 2 * 125
        else:
            self.in_dim = in_dim
            self.priv_in_dim = in_dim[1]
            self.publ_in_dim = in_dim[2]

        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_ff_layer = 1
        self.num_lstm_layer = num_lstm_layer

        ff_layers = [nn.Linear(self.priv_in_dim, self.hid_dim), nn.ReLU()]
        for i in range(1, self.num_ff_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(nn.ReLU())
        self.net = nn.Sequential(*ff_layers)

        self.lstm = nn.LSTM(
            self.hid_dim,
            self.hid_dim,
            num_layers=self.num_lstm_layer,
        ).to(device)
        self.lstm.flatten_parameters()

        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.out_dim)

        # for aux task
        self.pred_1st = nn.Linear(self.hid_dim, 5 * 3)

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, batchsize, self.hid_dim)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def act(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert priv_s.dim() == 2

        bsize = hid["h0"].size(0)
        assert hid["h0"].dim() == 4
        # hid size: [batch, num_layer, num_player, dim]
        # -> [num_layer, batch x num_player, dim]
        hid = {
            "h0": hid["h0"].transpose(0, 1).flatten(1, 2).contiguous(),
            "c0": hid["c0"].transpose(0, 1).flatten(1, 2).contiguous(),
        }

        priv_s = priv_s.unsqueeze(0)

        x = self.net(priv_s)
        o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        a = self.fc_a(o)
        a = a.squeeze(0)

        # hid size: [num_layer, batch x num_player, dim]
        # -> [batch, num_layer, num_player, dim]
        interim_hid_shape = (
            self.num_lstm_layer,
            bsize,
            -1,
            self.hid_dim,
        )
        h = h.view(*interim_hid_shape).transpose(0, 1)
        c = c.view(*interim_hid_shape).transpose(0, 1)

        return a, {"h0": h, "c0": c}

    @torch.jit.script_method
    def forward(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        action: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
            priv_s.dim() == 3 or priv_s.dim() == 2
        ), "dim = 3/2, [seq_len(optional), batch, dim]"

        one_step = False
        if priv_s.dim() == 2:
            priv_s = priv_s.unsqueeze(0)
            publ_s = publ_s.unsqueeze(0)
            legal_move = legal_move.unsqueeze(0)
            action = action.unsqueeze(0)
            one_step = True

        x = self.net(priv_s)
        if len(hid) == 0:
            o, _ = self.lstm(x)
        else:
            o, _ = self.lstm(x, (hid["h0"], hid["c0"]))
        a = self.fc_a(o)
        v = self.fc_v(o)
        q = duel(v, a, legal_move)

        # q: [seq_len, batch, num_action]
        # action: [seq_len, batch]
        qa = q.gather(2, action.unsqueeze(2)).squeeze(2)

        assert q.size() == legal_move.size()
        legal_q = (1 + q - q.min()) * legal_move
        # greedy_action: [seq_len, batch]
        greedy_action = legal_q.argmax(2).detach()

        if one_step:
            qa = qa.squeeze(0)
            greedy_action = greedy_action.squeeze(0)
            o = o.squeeze(0)
            q = q.squeeze(0)
        return qa, greedy_action, q, o

    def pred_loss_1st(self, lstm_o, target, hand_slot_mask, seq_len):
        return cross_entropy(self.pred_1st, lstm_o, target, hand_slot_mask, seq_len)


class PublicLSTMNet(torch.jit.ScriptModule):
    __constants__ = ["hid_dim", "out_dim", "num_lstm_layer"]

    def __init__(self, device, in_dim, hid_dim, out_dim, num_lstm_layer):
        super().__init__()
        # for backward compatibility
        if isinstance(in_dim, int):
            assert in_dim == 783
            self.in_dim = in_dim
            self.priv_in_dim = in_dim - 125  # priv = public info + player i's private info
            self.publ_in_dim = in_dim - 2 * 125
        else:
            self.in_dim = in_dim
            self.priv_in_dim = in_dim[1]
            self.publ_in_dim = in_dim[2]

        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_ff_layer = 1
        self.num_lstm_layer = num_lstm_layer

        self.priv_net = nn.Sequential(
            nn.Linear(self.priv_in_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
        )

        ff_layers = [nn.Linear(self.publ_in_dim, self.hid_dim), nn.ReLU()]
        for i in range(1, self.num_ff_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(nn.ReLU())
        self.publ_net = nn.Sequential(*ff_layers)

        self.lstm = nn.LSTM(
            self.hid_dim,
            self.hid_dim,
            num_layers=self.num_lstm_layer,
        ).to(device)
        self.lstm.flatten_parameters()

        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.out_dim)

        # for aux task
        self.pred_1st = nn.Linear(self.hid_dim, 5 * 3)

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, batchsize, self.hid_dim)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def act(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # priv_s:[batch, dim]
        # publ_s:[batch, dim]
        # hid:[batch, num_layer, num_player, dim]
        
        assert priv_s.dim() == 2

        bsize = hid["h0"].size(0)
        assert hid["h0"].dim() == 4
        # hid size: [batch, num_layer, num_player, dim]
        # -> [num_layer, batch x num_player, dim]
        hid = {
            "h0": hid["h0"].transpose(0, 1).flatten(1, 2).contiguous(),
            "c0": hid["c0"].transpose(0, 1).flatten(1, 2).contiguous(),
        }

        # priv_s:[1, batch, dim] : add the seq_len dim
        priv_s = priv_s.unsqueeze(0)
        # publ_s:[1, batch, dim]
        publ_s = publ_s.unsqueeze(0)

        # x:[1, batch, hid_dim]
        x = self.publ_net(publ_s)
        # publ_o:[1, batch, hid_dim]
        publ_o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))

        # priv_o:[1, batch, hid_dim]
        priv_o = self.priv_net(priv_s)
        # o:[1, batch, hid_dim]
        o = priv_o * publ_o
        a = self.fc_a(o)
        # a:[batch, out_dim]
        a = a.squeeze(0)

        # turn it back
        # hid size: [num_layer, batch x num_player, dim]
        # -> [batch, num_layer, num_player, dim]
        interim_hid_shape = (
            self.num_lstm_layer,
            bsize,
            -1,
            self.hid_dim,
        )
        h = h.view(*interim_hid_shape).transpose(0, 1)
        c = c.view(*interim_hid_shape).transpose(0, 1)

        return a, {"h0": h, "c0": c}

    @torch.jit.script_method
    def forward(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        action: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
            priv_s.dim() == 3 or priv_s.dim() == 2
        ), "dim = 3/2, [seq_len(optional), batch, dim]"

        one_step = False
        if priv_s.dim() == 2:
            priv_s = priv_s.unsqueeze(0)
            publ_s = publ_s.unsqueeze(0)
            legal_move = legal_move.unsqueeze(0)
            action = action.unsqueeze(0)
            one_step = True

        x = self.publ_net(publ_s)
        if len(hid) == 0:
            publ_o, _ = self.lstm(x)
        else:
            # x: [seq_len, batch, dim]
            # hid: [num_layer, batch*num_player, dim]
            
            # publ_o: [seq_len, batch, dim]
            publ_o, _ = self.lstm(x, (hid["h0"], hid["c0"]))
        # priv_o: [seq_len, batch, dim]
        priv_o = self.priv_net(priv_s)
        # o: [seq_len, batch, dim]
        o = priv_o * publ_o
        a = self.fc_a(o)
        v = self.fc_v(o)
        q = duel(v, a, legal_move)

        # q: [seq_len, batch, num_action]
        # action: [seq_len, batch]
        
        # qa: [seq_len, batch]
        qa = q.gather(2, action.unsqueeze(2)).squeeze(2)

        assert q.size() == legal_move.size()
        legal_q = (1 + q - q.min()) * legal_move
        # greedy_action: [seq_len, batch]
        greedy_action = legal_q.argmax(2).detach()

        if one_step:
            qa = qa.squeeze(0)
            greedy_action = greedy_action.squeeze(0)
            o = o.squeeze(0)
            q = q.squeeze(0)
        return qa, greedy_action, q, o

    def pred_loss_1st(self, lstm_o, target, hand_slot_mask, seq_len):
        return cross_entropy(self.pred_1st, lstm_o, target, hand_slot_mask, seq_len)


class PPOPublicLSTMNet(torch.jit.ScriptModule):
    __constants__ = ["hid_dim", "out_dim", "num_lstm_layer"]

    def __init__(self, device, in_dim, hid_dim, out_dim, num_lstm_layer, perfect):
        super().__init__()
        # for backward compatibility
        if isinstance(in_dim, int):
            assert in_dim == 783
            self.in_dim = in_dim
            self.priv_in_dim = in_dim - 125  # priv = public info + player i's private info
            self.publ_in_dim = in_dim - 2 * 125
            self.perfect_in_dim = in_dim
        else:
            self.in_dim = in_dim
            self.priv_in_dim = in_dim[1]
            self.publ_in_dim = in_dim[2]
            self.perfect_in_dim = in_dim[1] + 125

        self.perfect = perfect
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_ff_layer = 1
        self.num_lstm_layer = num_lstm_layer

        self.priv_net = nn.Sequential(
            nn.Linear(self.priv_in_dim, self.hid_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hid_dim),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hid_dim),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hid_dim),
        )

        ff_layers = [nn.Linear(self.publ_in_dim, self.hid_dim), nn.ReLU(), nn.LayerNorm(self.hid_dim),]
        for i in range(1, self.num_ff_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(nn.ReLU())
            ff_layers.append(nn.LayerNorm(self.hid_dim))
        self.publ_net = nn.Sequential(*ff_layers)

        self.lstm = nn.LSTM(
            self.hid_dim,
            self.hid_dim,
            num_layers=self.num_lstm_layer,
        ).to(device)
        self.lstm.flatten_parameters()

        self.policy = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hid_dim),
            nn.Linear(self.hid_dim, self.out_dim),
        )
        
        self.fc = nn.Linear(self.hid_dim, self.hid_dim)
        
        self.emb_i = nn.Sequential(
            nn.Linear(self.priv_in_dim, self.hid_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hid_dim),
        )
        
        self.emb_p = nn.Sequential(
            nn.Linear(self.perfect_in_dim, self.hid_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hid_dim),
        )
        
        self.value = nn.Sequential(
            nn.Linear(2 * self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hid_dim),
            nn.Linear(self.hid_dim, 1),
        )
        
        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.out_dim)
        # for aux task
        self.pred_1st = nn.Linear(self.hid_dim, 5 * 3)

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, batchsize, self.hid_dim)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def act(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        hid: Dict[str, torch.Tensor],
        legal_move: torch.Tensor,
        perf_s: torch.Tensor,
        training: Boolean
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        # priv_s:[batch, dim]
        # publ_s:[batch, dim]
        # hid:[batch, num_layer, num_player, dim]
        
        assert priv_s.dim() == 2

        bsize = hid["h0"].size(0)
        assert hid["h0"].dim() == 4
        # hid size: [batch, num_layer, num_player, dim]
        # -> [num_layer, batch x num_player, dim]
        hid = {
            "h0": hid["h0"].transpose(0, 1).flatten(1, 2).contiguous(),
            "c0": hid["c0"].transpose(0, 1).flatten(1, 2).contiguous(),
        }

        # priv_s:[1, batch, dim] : add the seq_len dim
        priv_s = priv_s.unsqueeze(0)
        # publ_s:[1, batch, dim]
        publ_s = publ_s.unsqueeze(0)
        
        if self.perfect:
            perf_s = perf_s.unsqueeze(0)

        # x:[1, batch, hid_dim]
        x = self.publ_net(publ_s)
        # publ_o:[1, batch, hid_dim]
        publ_o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))

        # priv_o:[1, batch, hid_dim]
        priv_o = self.priv_net(priv_s)
        # o:[1, batch, hid_dim]
        o = priv_o * publ_o
        o = self.fc(o)
        
        # probs: [1, batch, out_dim]
        probs = self.policy(o)
        legal_move = legal_move.unsqueeze(0)
        
        assert probs.size() == legal_move.size()
        assert legal_move.dim() == 3  # 1, batch, out_dim
        legal_probs = probs + (legal_move - 1) * 1e10
        # p: [batch, num_action]
        p = torch.softmax(legal_probs, dim=-1).squeeze(0)
        # greedy_action: [batch]
        greedy_action = p.argmax(1).detach()
    
        if training:
            # v_emb: [1, batch, dim]
            if self.perfect:
                v_emb = self.emb_p(perf_s)
            else:
                v_emb = self.emb_i(priv_s)
            # v_emb: [1, batch, 2*dim]
            v_in = torch.cat([o, v_emb], -1)
            # values: [1, batch, 1]
            values = self.value(v_in)
            # print(values.shape)
            # values: [batch]
            values = values.squeeze(-1).squeeze(0)
        else:
            values =  torch.zeros_like(greedy_action)

        # turn it back
        # hid size: [num_layer, batch x num_player, dim]
        # -> [batch, num_layer, num_player, dim]
        interim_hid_shape = (
            self.num_lstm_layer,
            bsize,
            -1,
            self.hid_dim,
        )
        h = h.view(*interim_hid_shape).transpose(0, 1)
        c = c.view(*interim_hid_shape).transpose(0, 1)

        return greedy_action, {"h0": h, "c0": c}, p, values

    @torch.jit.script_method
    def act_other(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # priv_s:[batch, dim]
        # publ_s:[batch, dim]
        # hid:[batch, num_layer, num_player, dim]
        
        assert priv_s.dim() == 2

        bsize = hid["h0"].size(0)
        assert hid["h0"].dim() == 4
        # hid size: [batch, num_layer, num_player, dim]
        # -> [num_layer, batch x num_player, dim]
        hid = {
            "h0": hid["h0"].transpose(0, 1).flatten(1, 2).contiguous(),
            "c0": hid["c0"].transpose(0, 1).flatten(1, 2).contiguous(),
        }

        # priv_s:[1, batch, dim] : add the seq_len dim
        priv_s = priv_s.unsqueeze(0)
        # publ_s:[1, batch, dim]
        publ_s = publ_s.unsqueeze(0)

        # x:[1, batch, hid_dim]
        x = self.publ_net(publ_s)
        # publ_o:[1, batch, hid_dim]
        publ_o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))

        # priv_o:[1, batch, hid_dim]
        priv_o = self.priv_net(priv_s)
        # o:[1, batch, hid_dim]
        o = priv_o * publ_o
        a = self.fc_a(o)
        # a:[batch, out_dim]
        a = a.squeeze(0)

        # turn it back
        # hid size: [num_layer, batch x num_player, dim]
        # -> [batch, num_layer, num_player, dim]
        interim_hid_shape = (
            self.num_lstm_layer,
            bsize,
            -1,
            self.hid_dim,
        )
        h = h.view(*interim_hid_shape).transpose(0, 1)
        c = c.view(*interim_hid_shape).transpose(0, 1)

        return a, {"h0": h, "c0": c}

    @torch.jit.script_method
    def forward(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        action: torch.Tensor,
        hid: Dict[str, torch.Tensor],
        perf_s: torch.Tensor,
        training: Boolean
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
            priv_s.dim() == 3 or priv_s.dim() == 2
        ), "dim = 3/2, [seq_len(optional), batch, dim]"

        one_step = False
        if priv_s.dim() == 2:
            priv_s = priv_s.unsqueeze(0)
            publ_s = publ_s.unsqueeze(0)
            legal_move = legal_move.unsqueeze(0)
            action = action.unsqueeze(0)
            if self.perfect:
                perf_s = perf_s.unsqueeze(0)
            one_step = True

        x = self.publ_net(publ_s)
        if len(hid) == 0:
            publ_o, _ = self.lstm(x)
        else:
            # x: [seq_len, batch, dim]
            # hid: [num_layer, batch*num_player, dim]
            
            # publ_o: [seq_len, batch, dim]
            publ_o, _ = self.lstm(x, (hid["h0"], hid["c0"]))
        # priv_o: [seq_len, batch, dim]
        priv_o = self.priv_net(priv_s)
        # o: [seq_len, batch, dim]
        o = priv_o * publ_o
        o = self.fc(o)
        
        # probs: [seq_len, batch, out_dim]
        probs = self.policy(o)
        
        assert probs.size() == legal_move.size()
        assert legal_move.dim() == 3  # seq, batch, out_dim
        legal_probs = probs + (legal_move - 1) * 1e10
        # p: [seq_len, batch, num_action]
        p = torch.softmax(legal_probs, dim=-1)
        # greedy_action: [seq_len, batch]
        greedy_action = p.argmax(2).detach()
        
        if training:
            # v_emb: [seq_len, batch, dim]
            if self.perfect:
                v_emb = self.emb_p(perf_s)
            else:
                v_emb = self.emb_i(priv_s)
            # v_emb: [seq_len, batch, 2*dim]
            v_in = torch.cat([o, v_emb], -1)
            # values: [seq_len, batch, 1]
            values = self.value(v_in)
            # values: [seq_len, batch]
            values = values.squeeze(-1)
        else:
            values = torch.zeros_like(greedy_action)
        
        if one_step:
            values = values.squeeze(0)
            greedy_action = greedy_action.squeeze(0)
            o = o.squeeze(0)
            p = p.squeeze(0)
        return values, greedy_action, p, o

    @torch.jit.script_method
    def forward_other(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        action: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
            priv_s.dim() == 3 or priv_s.dim() == 2
        ), "dim = 3/2, [seq_len(optional), batch, dim]"

        one_step = False
        if priv_s.dim() == 2:
            priv_s = priv_s.unsqueeze(0)
            publ_s = publ_s.unsqueeze(0)
            legal_move = legal_move.unsqueeze(0)
            action = action.unsqueeze(0)
            one_step = True

        x = self.publ_net(publ_s)
        if len(hid) == 0:
            publ_o, _ = self.lstm(x)
        else:
            # x: [seq_len, batch, dim]
            # hid: [num_layer, batch*num_player, dim]
            
            # publ_o: [seq_len, batch, dim]
            publ_o, _ = self.lstm(x, (hid["h0"], hid["c0"]))
        # priv_o: [seq_len, batch, dim]
        priv_o = self.priv_net(priv_s)
        # o: [seq_len, batch, dim]
        o = priv_o * publ_o
        a = self.fc_a(o)
        v = self.fc_v(o)
        q = duel(v, a, legal_move)

        # q: [seq_len, batch, num_action]
        # action: [seq_len, batch]
        
        # qa: [seq_len, batch]
        qa = q.gather(2, action.unsqueeze(2)).squeeze(2)

        assert q.size() == legal_move.size()
        legal_q = (1 + q - q.min()) * legal_move
        # greedy_action: [seq_len, batch]
        greedy_action = legal_q.argmax(2).detach()

        if one_step:
            qa = qa.squeeze(0)
            greedy_action = greedy_action.squeeze(0)
            o = o.squeeze(0)
            q = q.squeeze(0)
        return qa, greedy_action, q, o
    
    def pred_loss_1st(self, lstm_o, target, hand_slot_mask, seq_len):
        return cross_entropy(self.pred_1st, lstm_o, target, hand_slot_mask, seq_len)