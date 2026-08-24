Algebnas Ch. 8

ring, and Jacobson semisimple, by Corollary 8.35(ii). The factor module  $ J^{q}/J^{q+1} $ is an  $ (R/J) $-module; hence, by Corollary 8.43,  $ J^{q}/J^{q+1} $ is a semisimple module, and so it can be decomposed into a direct sum of (possibly infinitely many) simple  $ (R/J) $-modules. But there can be only finitely many summands, for every  $ (R/J) $-submodule of  $ J^{q}/J^{q+1} $ is necessarily an R-submodule, and  $ J^{q}/J^{q+1} $ has the DCC on R-submodules. Hence, there are simple  $ (R/J) $-modules  $ S_i $ with

$$
J ^ { q } / J ^ { q + 1 } = S _ { 1 } \oplus S _ { 2 } \oplus \cdots \oplus S _ { p }
$$

Throwing away one simple summand at a time yields a series of  $ J^q/J^{q+1} $ whose 1th factor module is

$$
( S _ { i } \oplus S _ { i + 1 } \oplus \cdots \oplus S _ { p } ) / ( S _ { i + 1 } \oplus \cdots \oplus S _ { p } ) \cong S _ { i }
$$

Now the simple  $ (R/J) $-module  $ S_i $ is also a simple  $ R $-module, for it is an  $ R $-module annihilated by  $ J $, so that we have constructed a composition series for  $ J^q/J^{q+1} $ as a left  $ R $-module. Finally, refine the original series for  $ R $ in this way, for every  $ q $, to obtain a composition series for  $ R $.

Of course, the converse of Theorem 8.46 is false.

The next result is fundamental.

Theorem 8.47 (Maschke's Theorem). If $G$ is a finite group and $k$ is a field whose characteristic does not divide $|G|$, then $kG$ is a left semisimple ring.

Remark. The hypothesis always holds if k has characteristic 0.

Proof. By Proposition 8.42, it suffices to prove that every left ideal $I$ of $kG$ is a direct summand. Since $k$ is a field, $kG$ is a vector space over $k$ and $I$ is a subspace. By Corollary 6.49, $I$ is a (vector space) direct summand: There is a subspace $V$ (which may not be a left ideal in $kG$) with $kG = I \oplus V$. There is a $k$-linear transformation $d: kG \to I$ with $d(b) = b$ for all $b \in I$ and with $ker d = V$ [each $u \in kG$ has a unique expression of the form $u = b + v$, where $b \in I$ and $v \in V$, and $d(u) = b$]. Were $d$ a $kG$-map, not merely a $k$-map, then we would be done, by the criterion of Corollary 7.17: $I$ is a summand of $kG$ if and only if it is a retract; that is, there is a $kG$-map $D: kG \to I$ with $D(u) = u$ for all $u \in I$. We now force $d$ to be a $kG$-map by an “averaging” process.

Define $D: kG \to kG$ by

$$
D ( u ) = \frac { 1 } { | G | } \sum _ { x \in G } x d ( x ^ { - 1 } u )
$$

for all  $ u \in kG $. Note that  $ |G| \neq 0 $ in  $ k $, by the hypothesis on the characteristic of  $ k $, and so it is invertible. It is obvious that  $ D $ is a  $ k $-map.

(i) im $D \subseteq I$.

If $u \in kG$ and $x \in G$, then $d(x^{-1}u) \in I$ (because $\operatorname{im} d \subseteq I$), and $xd(x^{-1}u) \in I$ because $I$ is a left ideal. Therefore, $D(u) \in I$, for each term in the defining sum of $D(u)$ lies in $I$.
