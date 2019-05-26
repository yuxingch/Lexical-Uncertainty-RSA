"""Skeleton adapted from Ling 130a/230a: Introduction to semantics and pragmatics,
Winter 2019 http://web.stanford.edu/class/linguist130a/
"""
import numpy as np
import pandas as pd


class RSA:
    """Implementation of the core Rational Speech Acts model.

    Parameters
    ----------
    lexicon : `np.array` or `pd.DataFrame`
        Messages along the rows, states along the columns.
    prior : array-like
        Same length as the number of colums in `lexicon`.
    costs : array-like
        Same length as the number of rows in `lexicon`.
    alpha : float
        The temperature parameter. Default: 1.0

    Parameters for the Extension
    ----------
    lexica: list of `np.array`, as defined in the paper Potts(2016)
        length is the number of possible worlds (eg. NN,NS,NA,SN,SS,SA,AN,AS,AA)
    lexica_prior: `np.array`
        P(L) as defined in the paper: prior distribution over lexica
    lexica_num: integer
    """
    # Extension
    def __init__(self, prior, costs, lexicon=None, mode=None, lexica=None, 
                 lexica_prior=None, lexica_num=None, alpha=1.0):
        self.lexicon = lexicon
        self.prior = np.array(prior)
        self.costs = np.array(costs)
        self.alpha = alpha

        # Extension:
        self.mode = mode
        if (self.mode is not None) and (lexica is None):
            raise Exception('Missing lexica information for lexical uncertainty model.')
        self.lexica = lexica
        self.lexica_prior = lexica_prior  # TODO: method to get lexica prior in general
        # if unspecified
        if self.mode is not None and self.lexica_prior is None:
            if lexica_num is None:
                raise Exception('lexica prior and number of lexica are both unspecified.')
            # if we know the number of uncertainty
            self.lexica_prior = np.array([1.0/lexica_num]*lexica_num)

    def literal_listener(self, a=None):
        """Literal listener predictions, which corresponds intuitively
        to truth conditions with priors.

        Extension:
            base mode - nothing changed, use the lexicon
            bergen mode - use the input argument a

        Returns
        -------
        np.array or pd.DataFrame, depending on `self.lexicon`.
        The rows correspond to messages, the columns to states.

        """
        if self.mode == None:  # base mode
            return rownorm(self.lexicon * self.prior)
        else:  # lexical uncertainty mode
            return rownorm(a * self.prior)

    def speaker(self, a=None):
        """Returns a matrix of pragmatic speaker predictions.

        Extension:
        Bergen(2012), eq.(6)(8)
        U_n = log(L_{n−1}(m|u))−c(u) if n > 1
        U_n = log(L_0(m|u,L))-c(u) if n = 1
        S_n = exp(lambda U_n)

        Returns
        -------
        np.array or pd.DataFrame, depending on `self.lexicon`.
        The rows correspond to states, the columns to states.
        """
        if self.mode == None:
            lit = self.literal_listener().T
        else:  # lexical uncertainty mode
            lit = a.T
        utilities = self.alpha * (safelog(lit) + self.costs)
        return rownorm(np.exp(utilities))

    def listener(self, a=None):
        """Returns a matrix of pragmatic listener predictions.

        Extension:
            Base model - nothing changed
            Bergen model - return L0(a)

        Returns
        -------
        np.array or pd.DataFrame, depending on `self.lexicon`.
        The rows correspond to messages, the columns to states.
        """
        if self.mode == None:
            spk = self.speaker().T
            return rownorm(spk * self.prior)
        else:
            return self.literal_listener(a.T)

    # Extension: Bergen
    def bergen_lexical_uncertainty_listener(self):
        """Returns a matrix of lexical uncertainty listener
        predictions based on Bergen's model:

        Bergen(2012), eq.(7)
        L_n = \sum_L P(m)P(L)Sn−1(u|m,L)
            where P(m) is the prior
            P(L) is the lexica prior
            Sn-1 is the speaker value (from previous layer)

        Returns
        -------
        """
        num_lexica = len(self.lexica)
        # s = self.speaker().T
        temp = []
        for i in range(num_lexica):
            curr_lexica = self.lexica[i]
            l_0 = self.literal_listener(curr_lexica)
            s_1 = self.speaker(l_0)
            temp.append((self.lexica_prior[i] * self.prior * s_1.T).values)
        temp_np = np.array(temp)
        L_n = np.sum(temp_np, axis=0)
        return rownorm(L_n)

    def run_lu_model(self):
        result = [self.bergen_lexical_uncertainty_listener()]
        result.append(self.speaker(result[0]))
        result.append(self.listener(result[1]))
        result.append(self.speaker(result[2]))
        result.append(self.listener(result[3]))
        result.append(self.speaker(result[4]))
        result.append(self.listener(result[5]))
        return result

def rownorm(mat):
    """Row normalization of np.array or pd.DataFrame"""
    return (mat.T / mat.sum(axis=1)).T


def safelog(vals):
    """Silence distracting warnings about log(0)."""
    with np.errstate(divide='ignore'):
        return np.log(vals)


if __name__ == '__main__':
    """Examples from the class handout."""

    from IPython.display import display

    def display_reference_game(mod):
        d = mod.lexicon.copy()
        d['costs'] = mod.costs
        d.loc['prior'] = list(mod.prior) + [""]
        d.loc['alpha'] = [mod.alpha] + [" "] * mod.lexicon.shape[1]
        display(d)

    # Core lexicon for base model:
    msgs = ['hat', 'glasses']
    states = ['r_1', 'r_2']
    lex = pd.DataFrame([
        [0.0, 1.0],
        [1.0, 1.0]], index=msgs, columns=states)

    print("="*70 + "\nEven priors and all-0 message costs\n")
    basic_mod = RSA(lexicon=lex, prior=[0.5, 0.5], costs=[-6.0, 0.0], alpha=1.0)

    display_reference_game(basic_mod)

    print("\nLiteral listener")
    display(basic_mod.literal_listener())

    print("\nPragmatic speaker")
    display(basic_mod.speaker())

    print("\nPragmatic listener")
    display(basic_mod.listener())

    # TODO: for lexical uncertainty model
    # (self, prior, costs, lexicon=None, mode=None, lexica=None, 
    # lexica_prior=None, lexica_num=None, alpha=1.0):
    def display_lu_reference_game(mod):
        for i in range(len(mod.lexica)):
            d = mod.lexica[i].copy()
            d['costs'] = mod.costs
            d.loc['prior'] = list(mod.prior) + [""]
            d.loc['alpha'] = [mod.alpha] + [" "] * mod.lexicon.shape[1]
            display(d)
    print("="*70 + "\nPlayer B scored/aced.\n")    
    msgs = ['scored', 'aced', '0']
    states = ['N', 'S', 'A']
    # case 1: scored: {<S,b>,<A,b>}
    lex_1 = pd.DataFrame([
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0]], index=msgs, columns=states)
    # case 2: scored: {<S,b>}
    lex_2 = pd.DataFrame([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0]], index=msgs, columns=states)
    # case 3: scored: {<A,b>}
    lex_3 = pd.DataFrame([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0]], index=msgs, columns=states)
    lex = [lex_1, lex_2, lex_3]
    bergen_mod = RSA(lexicon=lex_1, prior=[1.0/3, 1.0/3, 1.0/3], costs=[0.0, 0.0, -5.0], alpha=1.0,
                    mode='bergen', lexica_num=3, lexica=lex)
    # display_lu_reference_game(bergen_mod)
    result = bergen_mod.run_lu_model()
    d = pd.DataFrame(result[0], index=msgs, columns=states)
    display(d)
    d = pd.DataFrame(result[2], index=msgs, columns=states)
    display(d)
    d = pd.DataFrame(result[4], index=msgs, columns=states)
    display(d)
    d = pd.DataFrame(result[6], index=msgs, columns=states)
    display(d)

    print("="*70 + "\nPlayer A scored/aced.\n")
    msgs = ['A scored', 'A aced', 'B scored', 'B aced', 'some player scored', 'some player aced',\
        'every player scored', 'every player aced', 'no player scored', 'no player aced', '0']
    states = ['NN', 'NS', 'NA', 'SN', 'SS', 'SA', 'AN', 'AS', 'AA']
    lex = []
    # 1: scored: {<S,e>, <A,e>}, player A/B: {A, B}, some: at least some
    lex.append(pd.DataFrame([
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # A scored
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],  # A aced
        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],  # B scored
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],  # B aced
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # some player scored
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],  # some player aced
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],  # every player scored
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # every player aced
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # no player scored
        [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # no player aced
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], index=msgs, columns=states))
    # 2: scored: {<S,e>, <A,e>}, player A: only A (i.e. only A scored and etc.), some: at least some
    lex.append(pd.DataFrame([
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],  # A scored (only A scored)
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],  # A aced (only A aced)
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # B scored
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # B aced
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # some player scored
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],  # some player aced
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],  # every player scored
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # every player aced
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # no player scored
        [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # no player aced
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], index=msgs, columns=states))
    # 3: scored: {<S,e>, <A,e>}, player A/B: {A, B}, some: some but not all
    lex.append(pd.DataFrame([
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # A scored
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],  # A aced
        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],  # B scored
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],  # B aced
        [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # some player scored
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],  # some player aced
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],  # every player scored
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # every player aced
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # no player scored
        [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # no player aced
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], index=msgs, columns=states))
    # 4: scored: {<S,e>, <A,e>}, player A: only A (i.e. only A scored and etc.), some: but not all
    lex.append(pd.DataFrame([
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],  # A scored (only A scored)
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],  # A aced (only A aced)
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # B scored
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # B aced
        [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # some player scored
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],  # some player aced
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],  # every player scored
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # every player aced
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # no player scored
        [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # no player aced
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], index=msgs, columns=states))
    # 5: scored: {<S,e>}, player A/B: {A, B}, some: at least some
    lex.append(pd.DataFrame([
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],  # A scored
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],  # A aced
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],  # B scored
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],  # B aced
        [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],  # some player scored
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],  # some player aced
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # every player scored
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # every player aced
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],  # no player scored <---
        [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # no player aced
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], index=msgs, columns=states))
    # 6: scored: {<S,e>}, player A: only A (i.e. only A scored and etc.), some: at least some
    lex.append(pd.DataFrame([
        [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # A scored (only A scored)
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],  # A aced (only A aced)
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # B scored
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # B aced
        [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],  # some player scored
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],  # some player aced
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # every player scored
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # every player aced
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # no player scored
        [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # no player aced
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], index=msgs, columns=states))
    # 7: scored: {<S,e>}, player A/B: {A, B}, some: some but not all
    # ['NN', 'NS', 'NA', 'SN', 'SS', 'SA', 'AN', 'AS', 'AA']
    lex.append(pd.DataFrame([
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],  # A scored
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],  # A aced
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],  # B scored
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],  # B aced
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # some player scored
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],  # some player aced
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # every player scored
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # every player aced
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # no player scored
        [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # no player aced
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], index=msgs, columns=states))
    # 8: scored: {<S,e>}, player A: only A (i.e. only A scored and etc.), some: but not all
    lex.append(pd.DataFrame([
        [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # A scored (only A scored)
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],  # A aced (only A aced)
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # B scored
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # B aced
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],  # some player scored
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],  # some player aced
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # every player scored
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # every player aced
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],  # no player scored
        [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # no player aced
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], index=msgs, columns=states))
    
    bergen_mod = RSA(lexicon=lex[0], prior=[1.0/9]*9, costs=[0.0]*10+[-5.0], alpha=1.0,
                    mode='bergen', lexica_num=len(lex), lexica=lex)
    # display_lu_reference_game(bergen_mod)
    result = bergen_mod.run_lu_model()
    d = pd.DataFrame(result[2], index=msgs, columns=states)
    display(d)
