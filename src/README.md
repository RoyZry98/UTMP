# MulT model and Multimodal Sentiment Analysis Benchmarks

This directory contains the model architecture for Multimodal Transformer (MulT) as well as the three major multimodal sentiment analysis benchmarks we used in the paper. All datasets should be put in the `../data` folder (if it does not exist, use `mkdir` to create one). Depending on the dataset, we may have different sets of evaluation metrics (see `eval_metrics.py`).


[0.8151436796764503, 0.8110279215067696, 0.8195923535879793, 0.8276100079483856, 0.8150781922855949, 0.8280610883002288, 0.8187199023271619, 0.7944751996278198, 0.8186427721452436, 0.7902137851653468, 0.7681599133311996, 0.7842478894632797, 0.7595925711234941, 0.7565533708144744, 0.7571585150774558, 0.7529034771951088, 0.7382259512805178, 0.7588475522824513, 0.7515077383616298, 0.7722011868639955, 0.7712701116685541, 0.754631743525062, 0.7231929594397725, 0.753253874967716, 0.7466826557009071, 0.7098121017236892, 0.7039366158693483, 0.7048992857372302, 0.71108143390727, 0.7332608600307738, 0.7113979848556263, 0.701848642003965, 0.7130249518850562, 0.6968221779838092, 0.7037993419178082, 0.6884647971618902, 0.6931455196708846, 0.7104323729086405, 0.6977185272795615, 0.6867508105668892]

MAE:  0.6355679
Correlation Coefficient:  0.6500583987009617
mult_acc_7:  0.4794314021107043
mult_acc_5:  0.49170794744777085
F1 score:  0.8102387357806123
Accuracy:  0.8080110497237569


# Best MulT + Fed + MM local data
[0.8450099638302778, 0.8148423093494453, 0.7739799340210828, 0.723514002817036, 0.7377572555416698, 0.7095081505784695, 0.6990335270579509, 0.7196596843683332, 0.7189302432519107, 0.7168937529337552, 0.667167210602026, 0.7118366944130038, 0.7148589959386965, 0.6645892034721046, 0.6696796061123288, 0.6725505430354862, 0.7157972604581926, 0.6809524928165257, 0.6811192295506625, 0.6954553300980579, 0.6933548715308183, 0.6668520099691038, 0.647407453223957, 0.6654726787422742, 0.6609833797529394, 0.6929767920414796, 0.6886568272804459, 0.7078111898968401, 0.6819666454870589, 0.6630520952972595, 0.6539865547460958, 0.6368105911743839, 0.6555801229874378, 0.6444043811605635, 0.6351613393187805, 0.6515112950412616, 0.6542614506360608, 0.6530208938431951, 0.6441216911384829, 0.6454765579963896]
MAE:  0.63516134
Correlation Coefficient:  0.662217539540205
mult_acc_7:  0.4826620719362481
mult_acc_5:  0.4979539091104889
F1 score:  0.8144553158944676
Accuracy:  0.8107734806629834


# Best M-3Encoder + Fed + One Moda local data
MAE:  0.6418894
Correlation Coefficient:  0.6188645027789887
mult_acc_7:  0.4807236700409218
mult_acc_5:  0.4925694594012492
F1 score:  0.7731222903846369
Accuracy:  0.7687845303867403
# M-3Encoder + Fed + One Moda local data  (Un aligned text data)
MAE:  0.71053326
Correlation Coefficient:  0.5992840003740977
mult_acc_7:  0.46177040706439804
mult_acc_5:  0.46177040706439804
F1 score:  0.7877608001114094
Accuracy:  0.7696132596685082

# 20220621 Weight_3_channels M-3Encoder + Fed + One Moda local data  (Un aligned text data)
MAE:  0.64588624
Correlation Coefficient:  0.62154303596416
mult_acc_7:  0.47038552659918154
mult_acc_5:  0.47297006245961665
F1 score:  0.7913428955275701
Accuracy:  0.7886740331491713
--------------------------------------------------
[Local Model] Training language model
Epoch 40 | Client   0 | Time/Batch(ms) 713.56 | Train Loss 0.5692
[Local Model] Training audio model
Epoch 40 | Client   1 | Time/Batch(ms) 713.12 | Train Loss 0.7262
[Local Model] Training vision model
Epoch 40 | Client   2 | Time/Batch(ms) 708.36 | Train Loss 0.8288
--------------------------------------------------
Epoch 40 | Time 68.8520 sec | Valid Loss 0.6248 | Test Loss 0.6399
--------------------------------------------------
# 20220616: More Complex model update




[0.8285569488912365, 0.7855914982644335, 0.7688739987352196, 0.759791115312106, 0.7651560104101248, 0.7520459598311827, 0.7425993544619427, 0.7534111184522352, 0.7433368444070461, 0.7307041968453066]
MAE:  0.7307042
Correlation Coefficient:  0.5703205514813394
mult_acc_7:  0.4449709239715701
mult_acc_5:  0.4449709239715701
F1 score:  0.7821603893502388
Accuracy:  0.7776243093922652

# 20220708: Cross model + temperature
[0.8281200740092431, 0.7873691498475832, 0.7728801292256756, 0.756741231162564, 0.7511405343257633, 0.7410721494903498, 0.7303380915863412, 0.7329145569804412, 0.7198038352169971, 0.7158535563840502, 0.7171378910939498, 0.7092182525858242, 0.7100218040738259, 0.7026554120144093, 0.7064654512855223, 0.6937411771030126, 0.6955126476986203, 0.6958225849576996, 0.6918734358529348, 0.6907278707693446, 0.6685778384267218, 0.665910929987047, 0.6564755246810248, 0.6560276019503889, 0.6644275281133001, 0.6543442582983974, 0.6598428444130113, 0.6528000072456757, 0.6597093923349152, 0.6560663066494704, 0.6577365779039702, 0.6545230202214769, 0.6556956435158775, 0.6520370626511205, 0.6543531398975513, 0.6590191536285537, 0.653823116024378, 0.6533023171478269, 0.6513148866258457, 0.6472058880953035]
MAE:  0.6538231
Correlation Coefficient:  0.6206469793545808
mult_acc_7:  0.4622011630411372
mult_acc_5:  0.4622011630411372
F1 score:  0.7929581689718351
Accuracy:  0.7814917127071823
--------------------------------------------------

# 20220710 imcap temperature
[0.5448569263984908, 0.5408233277070751, 0.5421888890551098, 0.5416451795523101, 0.5407481709522988, 0.5401749868255689, 0.5385124971871691, 0.5372357617563276, 0.5346431538367322, 0.5343334430824719, 0.532214354096191, 0.5324402924285514, 0.5291306180105027, 0.5288264516320056, 0.5271560859832682, 0.5251253674914842, 0.5328030457882993, 0.5245151580778012, 0.5253006006990161, 0.5206519472065256, 0.5186477850622205, 0.5204451619180789, 0.5158963815362723, 0.515093507797225, 0.5210218472140176, 0.5137352719744132, 0.5116787400327003, 0.513709060164657, 0.5095715020765373, 0.5083170000042743, 0.5047885860715594, 0.504707401622333, 0.5063975455918546, 0.5043765891081234, 0.5118472394404381, 0.5049186284099815, 0.5033228689673613, 0.503194840477982, 0.5169158000935877, 0.5077348515423122]
Neutral:
  - F1 Score:  0.6111189465475279
  - Accuracy:  0.6396588486140725
Happy:
  - F1 Score:  0.7896951608577283
  - Accuracy:  0.8560767590618337
Sad:
  - F1 Score:  0.7026359623658505
  - Accuracy:  0.7931769722814499
Angry:
  - F1 Score:  0.7401882905558138
  - Accuracy:  0.7931769722814499

proj_l.weight
proj_a.weight
proj_v.weight
trans_l_with_a.version
trans_l_with_a.embed_positions._float_tensor
trans_l_with_a.layers.0.self_attn.in_proj_weight
trans_l_with_a.layers.0.self_attn.in_proj_bias
trans_l_with_a.layers.0.self_attn.out_proj.weight
trans_l_with_a.layers.0.self_attn.out_proj.bias
trans_l_with_a.layers.0.fc1.weight
trans_l_with_a.layers.0.fc1.bias
trans_l_with_a.layers.0.fc2.weight
trans_l_with_a.layers.0.fc2.bias
trans_l_with_a.layers.0.layer_norms.0.weight
trans_l_with_a.layers.0.layer_norms.0.bias
trans_l_with_a.layers.0.layer_norms.1.weight
trans_l_with_a.layers.0.layer_norms.1.bias
trans_l_with_a.layers.1.self_attn.in_proj_weight
trans_l_with_a.layers.1.self_attn.in_proj_bias
trans_l_with_a.layers.1.self_attn.out_proj.weight
trans_l_with_a.layers.1.self_attn.out_proj.bias
trans_l_with_a.layers.1.fc1.weight
trans_l_with_a.layers.1.fc1.bias
trans_l_with_a.layers.1.fc2.weight
trans_l_with_a.layers.1.fc2.bias
trans_l_with_a.layers.1.layer_norms.0.weight
trans_l_with_a.layers.1.layer_norms.0.bias
trans_l_with_a.layers.1.layer_norms.1.weight
trans_l_with_a.layers.1.layer_norms.1.bias
trans_l_with_a.layers.2.self_attn.in_proj_weight
trans_l_with_a.layers.2.self_attn.in_proj_bias
trans_l_with_a.layers.2.self_attn.out_proj.weight
trans_l_with_a.layers.2.self_attn.out_proj.bias
trans_l_with_a.layers.2.fc1.weight
trans_l_with_a.layers.2.fc1.bias
trans_l_with_a.layers.2.fc2.weight
trans_l_with_a.layers.2.fc2.bias
trans_l_with_a.layers.2.layer_norms.0.weight
trans_l_with_a.layers.2.layer_norms.0.bias
trans_l_with_a.layers.2.layer_norms.1.weight
trans_l_with_a.layers.2.layer_norms.1.bias
trans_l_with_a.layers.3.self_attn.in_proj_weight
trans_l_with_a.layers.3.self_attn.in_proj_bias
trans_l_with_a.layers.3.self_attn.out_proj.weight
trans_l_with_a.layers.3.self_attn.out_proj.bias
trans_l_with_a.layers.3.fc1.weight
trans_l_with_a.layers.3.fc1.bias
trans_l_with_a.layers.3.fc2.weight
trans_l_with_a.layers.3.fc2.bias
trans_l_with_a.layers.3.layer_norms.0.weight
trans_l_with_a.layers.3.layer_norms.0.bias
trans_l_with_a.layers.3.layer_norms.1.weight
trans_l_with_a.layers.3.layer_norms.1.bias
trans_l_with_a.layers.4.self_attn.in_proj_weight
trans_l_with_a.layers.4.self_attn.in_proj_bias
trans_l_with_a.layers.4.self_attn.out_proj.weight
trans_l_with_a.layers.4.self_attn.out_proj.bias
trans_l_with_a.layers.4.fc1.weight
trans_l_with_a.layers.4.fc1.bias
trans_l_with_a.layers.4.fc2.weight
trans_l_with_a.layers.4.fc2.bias
trans_l_with_a.layers.4.layer_norms.0.weight
trans_l_with_a.layers.4.layer_norms.0.bias
trans_l_with_a.layers.4.layer_norms.1.weight
trans_l_with_a.layers.4.layer_norms.1.bias
trans_l_with_a.layer_norm.weight
trans_l_with_a.layer_norm.bias
trans_l_with_v.version
trans_l_with_v.embed_positions._float_tensor
trans_l_with_v.layers.0.self_attn.in_proj_weight
trans_l_with_v.layers.0.self_attn.in_proj_bias
trans_l_with_v.layers.0.self_attn.out_proj.weight
trans_l_with_v.layers.0.self_attn.out_proj.bias
trans_l_with_v.layers.0.fc1.weight
trans_l_with_v.layers.0.fc1.bias
trans_l_with_v.layers.0.fc2.weight
trans_l_with_v.layers.0.fc2.bias
trans_l_with_v.layers.0.layer_norms.0.weight
trans_l_with_v.layers.0.layer_norms.0.bias
trans_l_with_v.layers.0.layer_norms.1.weight
trans_l_with_v.layers.0.layer_norms.1.bias
trans_l_with_v.layers.1.self_attn.in_proj_weight
trans_l_with_v.layers.1.self_attn.in_proj_bias
trans_l_with_v.layers.1.self_attn.out_proj.weight
trans_l_with_v.layers.1.self_attn.out_proj.bias
trans_l_with_v.layers.1.fc1.weight
trans_l_with_v.layers.1.fc1.bias
trans_l_with_v.layers.1.fc2.weight
trans_l_with_v.layers.1.fc2.bias
trans_l_with_v.layers.1.layer_norms.0.weight
trans_l_with_v.layers.1.layer_norms.0.bias
trans_l_with_v.layers.1.layer_norms.1.weight
trans_l_with_v.layers.1.layer_norms.1.bias
trans_l_with_v.layers.2.self_attn.in_proj_weight
trans_l_with_v.layers.2.self_attn.in_proj_bias
trans_l_with_v.layers.2.self_attn.out_proj.weight
trans_l_with_v.layers.2.self_attn.out_proj.bias
trans_l_with_v.layers.2.fc1.weight
trans_l_with_v.layers.2.fc1.bias
trans_l_with_v.layers.2.fc2.weight
trans_l_with_v.layers.2.fc2.bias
trans_l_with_v.layers.2.layer_norms.0.weight
trans_l_with_v.layers.2.layer_norms.0.bias
trans_l_with_v.layers.2.layer_norms.1.weight
trans_l_with_v.layers.2.layer_norms.1.bias
trans_l_with_v.layers.3.self_attn.in_proj_weight
trans_l_with_v.layers.3.self_attn.in_proj_bias
trans_l_with_v.layers.3.self_attn.out_proj.weight
trans_l_with_v.layers.3.self_attn.out_proj.bias
trans_l_with_v.layers.3.fc1.weight
trans_l_with_v.layers.3.fc1.bias
trans_l_with_v.layers.3.fc2.weight
trans_l_with_v.layers.3.fc2.bias
trans_l_with_v.layers.3.layer_norms.0.weight
trans_l_with_v.layers.3.layer_norms.0.bias
trans_l_with_v.layers.3.layer_norms.1.weight
trans_l_with_v.layers.3.layer_norms.1.bias
trans_l_with_v.layers.4.self_attn.in_proj_weight
trans_l_with_v.layers.4.self_attn.in_proj_bias
trans_l_with_v.layers.4.self_attn.out_proj.weight
trans_l_with_v.layers.4.self_attn.out_proj.bias
trans_l_with_v.layers.4.fc1.weight
trans_l_with_v.layers.4.fc1.bias
trans_l_with_v.layers.4.fc2.weight
trans_l_with_v.layers.4.fc2.bias
trans_l_with_v.layers.4.layer_norms.0.weight
trans_l_with_v.layers.4.layer_norms.0.bias
trans_l_with_v.layers.4.layer_norms.1.weight
trans_l_with_v.layers.4.layer_norms.1.bias
trans_l_with_v.layer_norm.weight
trans_l_with_v.layer_norm.bias
trans_l_with_l.version
trans_l_with_l.embed_positions._float_tensor
trans_l_with_l.layers.0.self_attn.in_proj_weight
trans_l_with_l.layers.0.self_attn.in_proj_bias
trans_l_with_l.layers.0.self_attn.out_proj.weight
trans_l_with_l.layers.0.self_attn.out_proj.bias
trans_l_with_l.layers.0.fc1.weight
trans_l_with_l.layers.0.fc1.bias
trans_l_with_l.layers.0.fc2.weight
trans_l_with_l.layers.0.fc2.bias
trans_l_with_l.layers.0.layer_norms.0.weight
trans_l_with_l.layers.0.layer_norms.0.bias
trans_l_with_l.layers.0.layer_norms.1.weight
trans_l_with_l.layers.0.layer_norms.1.bias
trans_l_with_l.layers.1.self_attn.in_proj_weight
trans_l_with_l.layers.1.self_attn.in_proj_bias
trans_l_with_l.layers.1.self_attn.out_proj.weight
trans_l_with_l.layers.1.self_attn.out_proj.bias
trans_l_with_l.layers.1.fc1.weight
trans_l_with_l.layers.1.fc1.bias
trans_l_with_l.layers.1.fc2.weight
trans_l_with_l.layers.1.fc2.bias
trans_l_with_l.layers.1.layer_norms.0.weight
trans_l_with_l.layers.1.layer_norms.0.bias
trans_l_with_l.layers.1.layer_norms.1.weight
trans_l_with_l.layers.1.layer_norms.1.bias
trans_l_with_l.layers.2.self_attn.in_proj_weight
trans_l_with_l.layers.2.self_attn.in_proj_bias
trans_l_with_l.layers.2.self_attn.out_proj.weight
trans_l_with_l.layers.2.self_attn.out_proj.bias
trans_l_with_l.layers.2.fc1.weight
trans_l_with_l.layers.2.fc1.bias
trans_l_with_l.layers.2.fc2.weight
trans_l_with_l.layers.2.fc2.bias
trans_l_with_l.layers.2.layer_norms.0.weight
trans_l_with_l.layers.2.layer_norms.0.bias
trans_l_with_l.layers.2.layer_norms.1.weight
trans_l_with_l.layers.2.layer_norms.1.bias
trans_l_with_l.layers.3.self_attn.in_proj_weight
trans_l_with_l.layers.3.self_attn.in_proj_bias
trans_l_with_l.layers.3.self_attn.out_proj.weight
trans_l_with_l.layers.3.self_attn.out_proj.bias
trans_l_with_l.layers.3.fc1.weight
trans_l_with_l.layers.3.fc1.bias
trans_l_with_l.layers.3.fc2.weight
trans_l_with_l.layers.3.fc2.bias
trans_l_with_l.layers.3.layer_norms.0.weight
trans_l_with_l.layers.3.layer_norms.0.bias
trans_l_with_l.layers.3.layer_norms.1.weight
trans_l_with_l.layers.3.layer_norms.1.bias
trans_l_with_l.layers.4.self_attn.in_proj_weight
trans_l_with_l.layers.4.self_attn.in_proj_bias
trans_l_with_l.layers.4.self_attn.out_proj.weight
trans_l_with_l.layers.4.self_attn.out_proj.bias
trans_l_with_l.layers.4.fc1.weight
trans_l_with_l.layers.4.fc1.bias
trans_l_with_l.layers.4.fc2.weight
trans_l_with_l.layers.4.fc2.bias
trans_l_with_l.layers.4.layer_norms.0.weight
trans_l_with_l.layers.4.layer_norms.0.bias
trans_l_with_l.layers.4.layer_norms.1.weight
trans_l_with_l.layers.4.layer_norms.1.bias
trans_l_with_l.layer_norm.weight
trans_l_with_l.layer_norm.bias
trans_l_mem.weight_ih_l0
trans_l_mem.weight_hh_l0
trans_l_mem.bias_ih_l0
trans_l_mem.bias_hh_l0
trans_l_mem.weight_ih_l1
trans_l_mem.weight_hh_l1
trans_l_mem.bias_ih_l1
trans_l_mem.bias_hh_l1
trans_l_mem.weight_ih_l2
trans_l_mem.weight_hh_l2
trans_l_mem.bias_ih_l2
trans_l_mem.bias_hh_l2
proj1.weight
proj1.bias
proj2.weight
proj2.bias
out_layer.weight
out_layer.bias