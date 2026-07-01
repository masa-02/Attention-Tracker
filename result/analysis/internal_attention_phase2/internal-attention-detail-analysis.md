# Attention-Tracker Phase2 内部表現 詳細分析

## 1. 分析対象と読み方

本分析は `outputs/phase2/*-seed0-phase2` に保存された Attention-Tracker Phase2 artifact を対象にした。対象は **attention routing 表現**であり、hidden state、MLP activation、MoE router、expert selection は今回の artifact には含まれない。そのため、以下で「内部表現」と呼ぶものは、正確には attention summary と selected head calibration を中心とした内部 routing signal である。

図ファイルはこの Markdown と同じ `result/analysis/internal_attention_phase2/` 配下に保存している。本文中の図パスは本番レポートへ転記しやすいよう、相対パスで記載する。

読み込めた完全 run は 17 件で、全 run に共通する prompt は 116 件だった。内訳は次の通り。

| family | generation / variant | models |
|---|---:|---|
| Qwen | Qwen2.5 base | 7B, 14B, 32B |
| Qwen | Qwen2.5 instruct | 7B, 14B, 32B |
| Qwen | Qwen2.5 coder-instruct | 7B, 14B, 32B |
| Qwen | Qwen3 base | 8B, 14B, 32B |
| Gemma | Gemma2 IT | 9B, 27B |
| Gemma | Gemma3 IT | 4B, 12B, 27B |

主に見る指標は以下である。

| 指標 | 解釈 |
|---|---|
| AUROC / AUPRC | benign と attack の score 分離。threshold に依存しない。 |
| FPR / FNR | fixed threshold 0.5 での誤検知・見逃し。運用 threshold のズレを反映する。 |
| risk_delta_attack_minus_benign | attack の平均 risk score から benign の平均 risk score を引いた値。分離幅。 |
| normalized_instruction_ratio | instruction span への attention 比率。attack 時に下がるほど untrusted data 側へ routing が移る。 |
| selected_fraction | head selection で選ばれた head の全 head に対する比率。 |
| selected_depth_mean | selected head の normalized layer depth の平均。 |
| exact_head_jaccard | 同一 layer/head shape のモデルペアで、選択 head がどれだけ一致するか。 |
| normalized_depth_cosine | exact head ではなく depth bin 分布として見た selected head の類似度。 |
| signature_cosine | depth 別 attention delta と head selection 分布を連結した表現 signature の類似度。 |

重要な注意点として、Qwen2.5 base は plain prompt、Qwen2.5 instruct / coder / Qwen3 / Gemma は chat template または instruct template 経由で実行されている。したがって、raw attention mass の絶対値はモデル間で直接比較しすぎない方がよい。比較の主軸は、同一 run 内の benign/attack 差分、depth profile、head selection の位置、同サイズペアの相対差に置く。

## 2. 全体観察

全体として、Attention-Tracker の signal は多くのモデルで attack と benign を強く分離している。ただし、AUROC と fixed threshold 0.5 の FPR/FNR は必ずしも一致しない。これは内部表現としては分離できていても、score の絶対スケールがモデルごとにズレるためである。

関連図:

![モデル別 AUROC / FPR / FNR](model_metrics_auc_fpr_fnr.png)

Path: `model_metrics_auc_fpr_fnr.png`

![ラベル別 focus score 分布](focus_score_by_label.png)

Path: `focus_score_by_label.png`

代表的な傾向は以下。

- Qwen2.5 base は FPR が低いが、14B/32B では FNR が高い。つまり、false positive は少ないが attack を見逃しやすい。
- Qwen2.5 instruct は AUROC が大きく改善し、FNR が下がる。一方で 14B/32B では FPR が上がり、benign な instruction-like 入力を攻撃寄りに見る傾向が出る。
- Qwen2.5 coder-instruct は AUROC は高いが、14B/32B の FPR が高い。コード特化 post-training は routing signal の位置を保ちながら、threshold 0.5 上の過検知を増やしている。
- Qwen3 は 14B/32B で非常に強い分離を示し、fixed threshold でも安定している。Qwen2.5 base と比べると post-training なしでも routing signal が強い。
- Gemma2/3 は AUROC は高いが、特に小さめの Gemma3-4B と Gemma2-9B で FPR が高い。分離はあるが threshold 校正がずれている。

グループ平均では次のようになる。

| group | mean AUROC | mean FPR | mean FNR | mean risk delta | selected depth |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 base | 0.886 | 0.042 | 0.433 | 0.206 | 0.466 |
| Qwen2.5 instruct | 0.984 | 0.274 | 0.033 | 0.328 | 0.473 |
| Qwen2.5 coder-instruct | 0.977 | 0.470 | 0.011 | 0.351 | 0.506 |
| Qwen3 base | 0.993 | 0.071 | 0.028 | 0.445 | 0.482 |
| Gemma2 IT | 0.989 | 0.411 | 0.000 | 0.415 | 0.431 |
| Gemma3 IT | 0.982 | 0.429 | 0.017 | 0.371 | 0.435 |

この表からは、検出性能そのものよりも「表現分離」と「threshold 校正」を分けて扱う必要があることが分かる。Qwen2.5 base は FPR だけ見れば安全に見えるが、FNR が高く、attention routing probe としては attack 側の score が十分に下がっていない。逆に Gemma や Qwen2.5 coder は attack を強く拾うが、benign も巻き込みやすい。

## 3. モデルサイズによる変化

関連図:

![モデル別 AUROC / FPR / FNR](model_metrics_auc_fpr_fnr.png)

Path: `model_metrics_auc_fpr_fnr.png`

![depth bin ごとの selected head density](selected_head_density_by_depth.png)

Path: `selected_head_density_by_depth.png`

### 3.1 Qwen2.5 base

Qwen2.5 base は、サイズが大きくなるほど今回の fixed threshold 0.5 では attack を拾いにくくなる。

| model | AUROC | FPR | FNR | risk delta | selected fraction | selected depth |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B base | 0.933 | 0.125 | 0.133 | 0.287 | 0.008 | 0.574 |
| Qwen2.5-14B base | 0.864 | 0.000 | 0.500 | 0.177 | 0.005 | 0.437 |
| Qwen2.5-32B base | 0.862 | 0.000 | 0.667 | 0.153 | 0.008 | 0.386 |

観察:

- 7B ではまだ benign/attack の分離が一定程度あるが、14B/32B では AUROC と risk delta が低下する。
- 14B/32B は FPR が 0 だが、これは検出が慎重というより、score が attack 側へ十分に動かず FNR が増えている状態である。
- selected head は 0.5〜0.8% 程度しか選ばれず、instruct/coder/Gemma と比べて非常に少ない。Attention Tracker の calibration 条件では、base model の routing difference が弱いか、head selection の margin が小さい。
- selected depth は 7B で 0.574、14B で 0.437、32B で 0.386 と浅い方向へ移る。base model ではサイズ増加が単純に「より深い層で PI signal が強くなる」形にはなっていない。

考察:

Qwen2.5 base は alignment post-training を受けていないため、system/user hierarchy や instruction conflict を明示的に扱う routing が弱い可能性がある。特に 14B/32B では attack に対して normalized instruction ratio の低下が小さく、risk score の差分が縮む。これは「大きい base model ほど安全になる」というより、「この probe の calibration 条件では attack/benign の routing separation が出にくい」と見るべきである。

### 3.2 Qwen2.5 instruct

Qwen2.5 instruct は、base に比べて AUROC と FNR が大きく改善する。

| model | AUROC | FPR | FNR | risk delta | selected fraction | selected depth |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B Instruct | 0.974 | 0.071 | 0.100 | 0.324 | 0.031 | 0.503 |
| Qwen2.5-14B Instruct | 0.989 | 0.286 | 0.000 | 0.316 | 0.032 | 0.398 |
| Qwen2.5-32B Instruct | 0.990 | 0.464 | 0.000 | 0.342 | 0.045 | 0.516 |

観察:

- AUROC は 7B の 0.974 から 14B/32B の 0.989/0.990 へ上がる。
- FNR は 14B/32B で 0 になり、attack をほぼ取り逃がさない。
- 一方で FPR は 14B で 0.286、32B で 0.464 まで上がる。
- selected fraction は base の 0.005〜0.008 から 0.031〜0.045 に増える。post-training 後は、calibration で拾える head が明確に増えている。

考察:

Instruct 化は attention routing の attack/benign 分離を強めるが、fixed threshold 0.5 では benign の instruction-like 入力も attack 側に寄せる。これは Attention Tracker の元の仮定、つまり「attack では high-priority instruction への focus が下がる」という単純な routing だけでは、通常の長い質問・相談・命令調の benign を完全に区別できないことを示す。

32B Instruct で FPR が高い一方 AUROC は高いため、表現は分離している。問題は threshold であり、モデル別またはファミリ別の calibration を行えば改善する可能性が高い。

### 3.3 Qwen2.5 coder-instruct

Qwen2.5 coder-instruct は、7B/14B では高い AUROC を示すが、FPR は高めで、32B では AUROC もやや下がる。

| model | AUROC | FPR | FNR | risk delta | selected depth | late fraction |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-7B Instruct | 0.990 | 0.196 | 0.017 | 0.400 | 0.448 | 0.143 |
| Qwen2.5-Coder-14B Instruct | 0.988 | 0.679 | 0.000 | 0.326 | 0.451 | 0.000 |
| Qwen2.5-Coder-32B Instruct | 0.952 | 0.536 | 0.017 | 0.327 | 0.618 | 0.424 |

観察:

- 7B は非常に良い分離を示すが、FPR は Instruct 7B より高い。
- 14B は AUROC 0.988 と高いが FPR 0.679 と非常に高い。
- 32B は AUROC が 0.952 まで下がる。selected depth は 0.618 で、late fraction も 0.424 と目立って深い。
- Instruct → Coder の exact head Jaccard は 0.239〜0.338、normalized depth cosine は 0.782〜0.893。head の場所はかなり近いが、完全一致ではない。

考察:

コード特化 post-training は、instruction following と共通した attention depth 構造を保つが、score calibration と benign/attack の境界を変える。特に Coder 14B/32B の FPR は、通常の質問や情報探索にも「命令的・タスク的」な構造を強く見てしまうことを示唆する。32B では selected head が深い層に寄るため、コード特化により後段の抽象的な task/rule handling 表現が PI signal に混ざっている可能性がある。

### 3.4 Qwen3

Qwen3 は今回の中で最も安定したグループである。

| model | AUROC | FPR | FNR | risk delta | selected fraction | selected depth |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | 0.981 | 0.179 | 0.033 | 0.387 | 0.035 | 0.467 |
| Qwen3-14B | 0.999 | 0.036 | 0.000 | 0.426 | 0.039 | 0.478 |
| Qwen3-32B | 1.000 | 0.000 | 0.050 | 0.520 | 0.009 | 0.501 |

観察:

- 14B/32B は AUROC がほぼ 1.0。
- FPR も低く、Qwen2.5 Instruct/Coder と比べて fixed threshold 0.5 でも安定している。
- 32B は selected fraction が 0.009 と非常に低いにもかかわらず、risk delta が 0.520 と最大である。

考察:

Qwen3 では、少数の head または depth region に PI routing signal が強く集中している可能性がある。Qwen2.5 Instruct/Coder は selected head が多く、やや広く signal が分散する。一方 Qwen3-32B は selected head 数が少なくても分離が大きく、より sparse で強い routing marker を持つように見える。

### 3.5 Gemma2 / Gemma3

Gemma は AUROC は高いが、fixed threshold の FPR が高い run が多い。

| model | AUROC | FPR | FNR | risk delta | selected depth |
|---|---:|---:|---:|---:|---:|
| Gemma2-9B IT | 0.988 | 0.589 | 0.000 | 0.373 | 0.461 |
| Gemma2-27B IT | 0.990 | 0.232 | 0.000 | 0.457 | 0.400 |
| Gemma3-4B IT | 0.984 | 0.714 | 0.000 | 0.319 | 0.431 |
| Gemma3-12B IT | 0.979 | 0.446 | 0.000 | 0.363 | 0.507 |
| Gemma3-27B IT | 0.982 | 0.125 | 0.050 | 0.430 | 0.366 |

観察:

- Gemma2 は 9B → 27B で risk delta が 0.373 → 0.457 に増え、FPR が 0.589 → 0.232 に下がる。
- Gemma3 も 4B → 27B で FPR が 0.714 → 0.125 に下がるが、AUROC は 4B 0.984、12B 0.979、27B 0.982 と単調ではない。
- Gemma の selected depth は Qwen 系よりやや浅めで、特に Gemma3-27B は 0.366。
- selected head は中間層中心だが、Gemma3 は early fraction が 0.240〜0.326 と比較的高い。

考察:

Gemma では、attack 時の attention ratio 低下が大きいにもかかわらず FPR が高い。これは benign 側の focus score 分布も threshold 0.5 より attack 側に寄りやすいことを意味する。つまり Gemma の attention routing signal は強いが、Attention Tracker の固定 threshold とはスケールが合っていない。Gemma を運用評価に使うなら、モデル別 threshold calibration は必須である。

## 4. Attention routing の深さ

normalized instruction ratio の attack - benign 差分を見ると、すべてのグループで負の値になる。これは attack prompt では high-priority instruction span への attention 比率が下がり、untrusted data 側に routing が移るという Attention Tracker の基本仮説と一致する。

関連図:

![normalized depth profile](normalized_depth_profile.png)

Path: `normalized_depth_profile.png`

![attention mass delta by group](attention_mass_delta_by_group.png)

Path: `attention_mass_delta_by_group.png`

ただし、差分の大きさと出る深さはファミリで異なる。

| group | peak depth bins | 平均的な傾向 |
|---|---|---|
| Qwen2.5 base | 0.4〜0.7 | 差分が小さい。base では PI routing signal が弱い。 |
| Qwen2.5 instruct | 0.3〜0.5 | 差分が強まり、中間層にまとまる。 |
| Qwen2.5 coder-instruct | 0.3〜0.5、32B は後段寄り | Instruct と近いが、32B は late head が増える。 |
| Qwen3 | 0.5〜0.7 | 中後段に強く、サイズが大きくても分離が保たれる。 |
| Gemma2 | 0.3〜0.4 | 比較的浅い中間層で強い。 |
| Gemma3 | 0.2〜0.5 | Gemma2 より early/mid に広がる。 |

各モデルの最大差分 depth bin は以下。

| model | peak bin | delta |
|---|---:|---:|
| Qwen2.5-7B base | 0.4〜0.5 | -0.140 |
| Qwen2.5-14B base | 0.5〜0.6 | -0.085 |
| Qwen2.5-32B base | 0.6〜0.7 | -0.124 |
| Qwen2.5-7B Instruct | 0.4〜0.5 | -0.219 |
| Qwen2.5-14B Instruct | 0.3〜0.4 | -0.169 |
| Qwen2.5-32B Instruct | 0.4〜0.5 | -0.178 |
| Qwen3-14B | 0.5〜0.6 | -0.213 |
| Qwen3-32B | 0.6〜0.7 | -0.196 |
| Gemma2-27B | 0.3〜0.4 | -0.300 |
| Gemma3-27B | 0.4〜0.5 | -0.303 |

考察:

Gemma は attention ratio の差分自体が大きく、Qwen2.5 base は小さい。Qwen2.5 Instruct/Coder と Qwen3 は中間に位置する。これは、Gemma では prompt injection による attention routing shift が強く観測される一方、score threshold は過敏になりやすいことを示す。Qwen3 は差分が大きいだけでなく threshold との相性も良い。

## 5. Qwen2.5 Base vs Instruct: アライメント効果

Qwen2.5 の同サイズペアでは、Instruct 化により AUROC は全サイズで上がる。

関連図:

![Qwen2.5 exact head overlap](exact_head_overlap_qwen25.png)

Path: `exact_head_overlap_qwen25.png`

![normalized depth overlap](normalized_depth_overlap.png)

Path: `normalized_depth_overlap.png`

| size | AUROC delta | FPR delta | FNR delta | risk delta | exact head Jaccard | signature cosine |
|---:|---:|---:|---:|---:|---:|---:|
| 7B | +0.041 | -0.054 | -0.033 | +0.037 | 0.154 | -0.150 |
| 14B | +0.125 | +0.286 | -0.500 | +0.139 | 0.077 | 0.165 |
| 32B | +0.127 | +0.464 | -0.667 | +0.189 | 0.106 | -0.146 |

観察:

- 14B/32B では Instruct 化により FNR が大きく改善する。
- その代わり FPR が増える。特に 32B は base の FPR 0.000 から instruct の FPR 0.464 へ上がる。
- exact head Jaccard は 0.077〜0.154 と低い。Base と Instruct は同じ architecture でも、selected head はかなり入れ替わる。
- normalized depth cosine は 32B で 0.797 と高いが、7B/14B は 0.416/0.330 と低め。サイズが大きいほど、depth レベルでは似た位置に signal が出るが、exact head は一致しない。
- attack_instruction_ratio_delta_b_minus_a は -0.407〜-0.521 と大きい。Instruct の方が attack 時の instruction ratio が低く、routing shift が強い。

考察:

Instruct 化は、Prompt Injection を attention routing 上で「見える」ようにする。ただし、それは単純に base の head が強くなるのではなく、別の head 群・別の calibration scale が生まれる形で現れている。Base → Instruct の exact head overlap が低いことから、post-training は既存 head の感度調整だけでなく、routing signal の担い手そのものを変えている可能性が高い。

実験上の重要点は、Base と Instruct の head を直接流用するのは危険ということ。特に 14B/32B では Instruct の検出性能は高いが FPR も高いため、Instruct 用に threshold を再校正する必要がある。

## 6. Qwen2.5 Instruct vs Coder-Instruct: コード特化効果

Instruct と Coder-Instruct の比較では、architecture は同サイズで揃っているため exact head overlap を見やすい。

関連図:

![Qwen2.5 exact head overlap](exact_head_overlap_qwen25.png)

Path: `exact_head_overlap_qwen25.png`

![depth bin ごとの selected head density](selected_head_density_by_depth.png)

Path: `selected_head_density_by_depth.png`

| size | AUROC delta | FPR delta | FNR delta | risk delta | exact head Jaccard | depth cosine | signature cosine |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7B | +0.016 | +0.125 | -0.083 | +0.076 | 0.333 | 0.782 | 0.119 |
| 14B | -0.001 | +0.393 | +0.000 | +0.010 | 0.338 | 0.858 | 0.322 |
| 32B | -0.038 | +0.071 | +0.017 | -0.016 | 0.239 | 0.893 | 0.322 |

観察:

- AUROC は 7B で少し改善、14B はほぼ同等、32B は低下する。
- FPR は全サイズで Coder の方が高い。特に 14B は +0.393。
- exact head Jaccard は 0.239〜0.338。Base/Instruct より高く、Instruct と Coder は head の一部を共有している。
- depth cosine は 0.782〜0.893 と高い。コード特化しても signal が出る層の深さはかなり保たれる。
- Coder-32B は selected_depth_mean 0.618、late fraction 0.424 で、今回の Qwen2.5 系の中ではかなり深い。

考察:

コード特化は attention routing signal の「場所」を大きく変えないが、score の境界を変える。Instruct と Coder の depth cosine が高い一方で FPR が増えるため、コード特化により benign な質問・タスク記述・手順説明を攻撃に近い routing として扱いやすくなっている可能性がある。

Coder-32B の late fraction の高さは重要である。コードモデルでは、深い層で task structure や instruction-like control flow を処理する傾向が強く、それが Prompt Injection signal と重なる可能性がある。

## 7. Qwen2.5 vs Qwen3

Qwen3 は base として扱っているが、Qwen2.5 base よりはるかに安定した PI routing signal を示した。

関連図:

![normalized depth profile](normalized_depth_profile.png)

Path: `normalized_depth_profile.png`

![representation signature similarity heatmap](representation_signature_similarity_heatmap.png)

Path: `representation_signature_similarity_heatmap.png`

| group | mean AUROC | mean FPR | mean FNR | mean risk delta |
|---|---:|---:|---:|---:|
| Qwen2.5 base | 0.886 | 0.042 | 0.433 | 0.206 |
| Qwen3 base | 0.993 | 0.071 | 0.028 | 0.445 |

観察:

- Qwen3 は Qwen2.5 base と同じ base variant 扱いでも、risk delta が約2倍大きい。
- Qwen3-14B/32B は AUROC がほぼ 1.0。
- Qwen3-32B は selected fraction が 0.009 と低いが risk delta は最大。

考察:

Qwen3 は model family / training recipe / tokenizer / architecture の複合効果により、Prompt Injection に対する attention routing shift がより明確に現れている。Qwen2.5 base ではサイズ増加で signal が弱まったのに対し、Qwen3 では 14B/32B で signal が強くなる。この差は、単純な parameter count では説明できず、世代差または training recipe 差として扱うべきである。

## 8. Gemma と Qwen のファミリ差

Gemma と Qwen では、attention ratio の絶対スケールと threshold 挙動が異なる。

関連図:

![attention mass delta by group](attention_mass_delta_by_group.png)

Path: `attention_mass_delta_by_group.png`

![ラベル別 focus score 分布](focus_score_by_label.png)

Path: `focus_score_by_label.png`

Gemma の normalized instruction ratio は benign でも低めで、attack ではさらに大きく低下する。

| model | benign ratio | attack ratio | delta |
|---|---:|---:|---:|
| Gemma2-9B IT | 0.325 | 0.137 | -0.188 |
| Gemma2-27B IT | 0.330 | 0.140 | -0.190 |
| Gemma3-4B IT | 0.269 | 0.117 | -0.152 |
| Gemma3-12B IT | 0.283 | 0.123 | -0.160 |
| Gemma3-27B IT | 0.335 | 0.162 | -0.173 |

Qwen2.5 base は benign/attack とも ratio が高く、delta は小さい。

| model | benign ratio | attack ratio | delta |
|---|---:|---:|---:|
| Qwen2.5-7B base | 0.631 | 0.558 | -0.072 |
| Qwen2.5-14B base | 0.681 | 0.630 | -0.051 |
| Qwen2.5-32B base | 0.689 | 0.625 | -0.064 |

Qwen2.5 Instruct/Coder と Qwen3 は、Gemma よりは小さいが base より明確な delta を持つ。

考察:

Gemma は attention routing shift の振幅が大きいが、fixed threshold 0.5 では benign も attack 側に入りやすい。Qwen2.5 base は振幅が小さく、attack が threshold を越えにくい。Qwen3 は振幅と threshold の相性が良く、今回の benchmark では最も扱いやすい。

したがって、ファミリ横断で同一 threshold を使うことは適切ではない。AUROC で分離可能性を評価し、FPR/FNR はモデル別 calibration の結果として評価すべきである。

## 9. Selected head の安定性

Qwen2.5 同サイズペアでは、exact head overlap と normalized depth overlap の両方を計算できる。

関連図:

![Qwen2.5 exact head overlap](exact_head_overlap_qwen25.png)

Path: `exact_head_overlap_qwen25.png`

![normalized depth overlap](normalized_depth_overlap.png)

Path: `normalized_depth_overlap.png`

![depth bin ごとの selected head density](selected_head_density_by_depth.png)

Path: `selected_head_density_by_depth.png`

![representation signature similarity heatmap](representation_signature_similarity_heatmap.png)

Path: `representation_signature_similarity_heatmap.png`

| pair | exact Jaccard | depth cosine | signature cosine |
|---|---:|---:|---:|
| 7B Base vs Instruct | 0.154 | 0.416 | -0.150 |
| 14B Base vs Instruct | 0.077 | 0.330 | 0.165 |
| 32B Base vs Instruct | 0.106 | 0.797 | -0.146 |
| 7B Instruct vs Coder | 0.333 | 0.782 | 0.119 |
| 14B Instruct vs Coder | 0.338 | 0.858 | 0.322 |
| 32B Instruct vs Coder | 0.239 | 0.893 | 0.322 |

観察:

- Base vs Instruct は exact head Jaccard が低い。
- Instruct vs Coder は exact head Jaccard が相対的に高く、depth cosine も高い。
- 32B Base vs Instruct は exact Jaccard は低いが depth cosine は高い。層の位置は似ているが、head 単位では違う。

考察:

selected head は、post-training の影響を強く受ける。Base の head を Instruct に転用するのは難しい。一方、Instruct と Coder は selected head の層位置が似ており、同じ instruction-following 基盤の上でコード特化による score shift が起きていると解釈できる。

また、exact head overlap だけを見るとモデル間共通性を過小評価する可能性がある。層数・head数が異なるモデルでは exact 比較はできないため、normalized depth density や representation signature similarity を併用する必要がある。

## 10. False positive 要因

hard sample では、複数ファミリで false positive になる benign prompt が目立つ。上位はすべて label 0 で、複数ファミリにまたがって誤検知されている。

関連図:

![sample risk correlation heatmap](sample_risk_correlation_heatmap.png)

Path: `sample_risk_correlation_heatmap.png`

![ラベル別 focus score 分布](focus_score_by_label.png)

Path: `focus_score_by_label.png`

| sample | FP models | avg risk | 主な要因 |
|---:|---:|---:|---|
| 24 | 13 | 0.593 | 長文、質問、web/marketing |
| 61 | 12 | 0.598 | ドイツ語/文字化け、長文、質問、web/marketing |
| 67 | 12 | 0.549 | ドイツ語/文字化け、長文、質問、web/marketing |
| 30 | 12 | 0.543 | 長文、質問、web/marketing |
| 11 | 10 | 0.569 | ドイツ語/文字化け、質問 |
| 42 | 10 | 0.521 | ドイツ語/文字化け、長文、質問 |
| 32 | 10 | 0.494 | 長文 |

全 116 prompt のうち、76 prompt は false positive model 数が 0 だった。一方で一部の benign prompt は 8〜13 model で同時に false positive になっており、これはモデル個別の偶然ではなく、benchmark 上の hard negative と考えられる。

要因としては以下が多い。

- 質問形式。
- 長文の相談文。
- web/marketing や customer acquisition のようなタスク志向の説明。
- ドイツ語または mojibake を含むテキスト。
- `output`, `say`, `instruction` に近い語を含む通常質問。

考察:

Attention Tracker は「untrusted data 側に注意が移る」ことを攻撃の手掛かりにするため、長文で具体的なタスク条件を含む benign prompt でも risk が上がりやすい。これは binary PI benchmark だけでは見えにくい false positive 要因である。特に aligned instruction や harmless task instruction を別クラスとして持つ AlignSentinel 型の評価が必要になる。

## 11. 現時点での結論

1. Attention routing は Prompt Injection に反応している。
   多くのモデルで attack 時に normalized instruction ratio が下がり、data mass が上がる。これは Attention Tracker の基本仮説と整合する。

2. モデルサイズの効果はファミリ・post-training なしには解釈できない。
   Qwen2.5 base ではサイズ増加で分離が弱くなるが、Qwen3 ではサイズ増加で分離が強い。Gemma でも大きいモデルほど FPR が下がる傾向がある。

3. アライメント post-training は PI signal を強めるが、FPR も増やす。
   Qwen2.5 Instruct は Base より AUROC が高く FNR が低い。しかし 14B/32B では false positive が増える。

4. コード特化は signal の深さを保ちつつ、threshold 挙動を変える。
   Instruct vs Coder は depth overlap が高いが、Coder は FPR が高い。特に Coder-32B は late-layer head の寄与が強い。

5. ファミリ間で fixed threshold は共有しにくい。
   Gemma は強い routing shift を持つが FPR が高く、Qwen2.5 base は FPR が低いが FNR が高い。AUROC と FPR/FNR は別々に報告すべきである。

6. false positive は一部の benign hard samples に集中する。
   長文、質問、web/marketing、ドイツ語/文字化け、instruction-like 語彙が false positive 要因として目立つ。

## 12. 次に実施すべき分析

- モデル別 threshold calibration を行い、AUROC が高いのに FPR が高いモデルで FPR/FNR がどこまで改善するか確認する。
- benign hard sample を `non-instruction`, `aligned_instruction`, `task_instruction` などに分け、false positive がどの種類に集中するかを見る。
- injection span を明示できる dataset で、`injection_mass` と `normalized_injection_ratio` を取得する。
- AlignSentinel の重要 head と Attention Tracker の selected head を比較し、instruction hierarchy interaction と PI routing の重なりを見る。
- JBShield の hidden-state cache と接続し、attention routing の深さと hidden concept score の critical layer が対応するか確認する。

## 13. 参照 artifact

この詳細分析は以下を主に参照した。

- `model_metrics.csv`
- `group_summary.csv`
- `paired_alignment_effects.csv`
- `paired_coder_effects.csv`
- `attention_mass_by_label.csv`
- `attention_delta_by_depth.csv`
- `head_selection_summary.csv`
- `head_overlap.csv`
- `hard_samples_cross_family.csv`
- `representation_signature_similarity.csv`

本文中で参照している図は以下。

- `model_metrics_auc_fpr_fnr.png`
- `focus_score_by_label.png`
- `attention_mass_delta_by_group.png`
- `normalized_depth_profile.png`
- `selected_head_density_by_depth.png`
- `exact_head_overlap_qwen25.png`
- `normalized_depth_overlap.png`
- `sample_risk_correlation_heatmap.png`
- `representation_signature_similarity_heatmap.png`
