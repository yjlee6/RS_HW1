"""BPR with an explicit L2 regularization term (reg_lambda).

RecBole's built-in BPR only optimizes the BPR loss, so reg_lambda is added here:
    loss = BPRLoss(pos, neg) + reg_lambda * EmbLoss(u, i_pos, i_neg)
"""
from recbole.model.general_recommender.bpr import BPR
from recbole.model.loss import EmbLoss


class BPRReg(BPR):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.reg_lambda = config["reg_lambda"]
        self.reg_loss = EmbLoss()

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_score = (user_e * pos_e).sum(dim=1)
        neg_score = (user_e * neg_e).sum(dim=1)

        bpr_loss = self.loss(pos_score, neg_score)
        reg_loss = self.reg_loss(user_e, pos_e, neg_e)
        return bpr_loss + self.reg_lambda * reg_loss
