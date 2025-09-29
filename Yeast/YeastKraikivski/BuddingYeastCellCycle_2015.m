function [T,Y,yinit,param, allNames, allValues] = BuddingYeastCellCycle_2015(argTimeSpan,argYinit,argParam)
% [T,Y,yinit,param] = BuddingYeastCellCycle_2015(argTimeSpan,argYinit,argParam)
%
% input:
%     argTimeSpan is a vector of start and stop times (e.g. timeSpan = [0 10.0])
%     argYinit is a vector of initial conditions for the state variables (optional)
%     argParam is a vector of values for the parameters (optional)
%
% output:
%     T is the vector of times
%     Y is the vector of state variables
%     yinit is the initial conditions that were used
%     param is the parameter vector that was used
%     allNames is the output solution variable names
%     allValues is the output solution variable values corresponding to the names
%
%     example of running this file: [T,Y,yinit,param,allNames,allValues] = myMatlabFunc; <-(your main function name)
%

%
% Default time span
%
timeSpan = [0.0 1.0];

% output variable lengh and names
numVars = 150;
allNames = {'CDC5A';'CDC55';'UDNA';'PsS6S4p';'SPNALIGN';'CDC15';'PDS1T';'PsS6S4Wp';'CLN3';'CLN2';'BUB2';'PsS6Mp';'NET1A';'PsS6S4np';'APCP';'MASS';'SWI5T';'ORIFLAG';'BCK2';'CLB5T';'PsS4p';'SWE1T';'SWE1P';'ORI';'CKIT';'CKIP';'CLB2T';'SPN';'PmS6Mp';'MAD2A';'TEM1';'CDC20T';'BUD';'CDC14c';'CDH1A';'PmS6S4p';'CDC5T';'Wclb2';'CLB2f';'CLB2';'Wswe1';'ESP1';'GLUE';'I_SPOC';'S6S4';'PsS6S4W';'PsS6S4free';'PsS6S4';'S6S4free';'PmS6S4';'S4free';'LumpedJ_reaction_53';'LumpedJ_reaction_52';'LumpedJ_reaction_51';'VS2';'LumpedJ_reaction_50';'S6M';'PmS6M';'S6Mfree';'Wmcm1';'MCM1';'CDC14';'Wbub2';'LumpedJ_reaction_46';'Wpolo';'LumpedJ_reaction_45';'LumpedJ_reaction_44';'LumpedJ_reaction_43';'Wtem1';'LumpedJ_reaction_42';'Wcdc15';'LumpedJ_reaction_41';'CDC15f';'MEN';'CDC14n';'Wnet1';'LumpedJ_reaction_40';'Psfree';'LumpedJ_reaction_39';'LumpedJ_reaction_38';'Wcdc55';'LumpedJ_reaction_37';'CDC20A';'CDC20A_APCP';'CDC20A_APC';'LumpedJ_reaction_36';'Pmfree';'VS5';'LumpedJ_reaction_35';'CLB5';'Wcdh1';'SAC';'Wmad2';'LumpedJ_reaction_34';'LumpedJ_reaction_33';'LumpedJ_reaction_32';'Wapc';'LumpedJ_reaction_31';'LumpedJ_reaction_30';'Wssa1';'LumpedJ_reaction_29';'LumpedJ_reaction_28';'LumpedJ_reaction_27';'LumpedJ_reaction_26';'LumpedJ_reaction_25';'PsS4';'LumpedJ_reaction_24';'LumpedJ_reaction_23';'SSA1p';'Ydj1';'SSA1free';'CLN3n';'LumpedJ_reaction_22';'LumpedJ_reaction_21';'LumpedJ_reaction_20';'SPNvsESP';'Ws6s4w';'LumpedJ_reaction_19';'LumpedJ_reaction_18';'LumpedJ_reaction_17';'Wcki';'LumpedJ_reaction_16';'LumpedJ_reaction_15';'Wswi5';'SWI5A';'LumpedJ_reaction_14';'LumpedJ_reaction_13';'LumpedJ_reaction_12';'Ws4';'LumpedJ_reaction_11';'Ws6m';'Psfree2';'PsS6M';'LumpedJ_reaction_10';'Ws6s4np';'Whi5free';'mu';'UDNA1';'PmS4';'LumpedJ_reaction_9';'LumpedJ_reaction_8';'Ws6s4';'LumpedJ_reaction_7';'LumpedJ_reaction_6';'LumpedJ_reaction_5';'LumpedJ_reaction_4';'LumpedJ_reaction_3';'LumpedJ_reaction_2';'LumpedJ_reaction_1';'LumpedJ_reaction_0';};

if nargin >= 1
	if length(argTimeSpan) > 0
		%
		% TimeSpan overridden by function arguments
		%
		timeSpan = argTimeSpan;
	end
end
%
% Default Initial Conditions
%
yinit = [
	0.504749296263001;		% yinit(1) is the initial condition for 'CDC5A'
	0.027;		% yinit(2) is the initial condition for 'CDC55'
	0.0;		% yinit(3) is the initial condition for 'UDNA'
	0.0;		% yinit(4) is the initial condition for 'PsS6S4p'
	0.0;		% yinit(5) is the initial condition for 'SPNALIGN'
	0.859443593213341;		% yinit(6) is the initial condition for 'CDC15'
	0.0337966937259905;		% yinit(7) is the initial condition for 'PDS1T'
	0.136080572502587;		% yinit(8) is the initial condition for 'PsS6S4Wp'
	0.0840482413806777;		% yinit(9) is the initial condition for 'CLN3'
	0.196030848449198;		% yinit(10) is the initial condition for 'CLN2'
	0.0243221073519805;		% yinit(11) is the initial condition for 'BUB2'
	0.0;		% yinit(12) is the initial condition for 'PsS6Mp'
	0.275187325794182;		% yinit(13) is the initial condition for 'NET1A'
	0.0;		% yinit(14) is the initial condition for 'PsS6S4np'
	1.13738570370057;		% yinit(15) is the initial condition for 'APCP'
	1.0850756320167;		% yinit(16) is the initial condition for 'MASS'
	0.303096279050485;		% yinit(17) is the initial condition for 'SWI5T'
	1.0;		% yinit(18) is the initial condition for 'ORIFLAG'
	0.269830552689792;		% yinit(19) is the initial condition for 'BCK2'
	0.0380991441945013;		% yinit(20) is the initial condition for 'CLB5T'
	0.0;		% yinit(21) is the initial condition for 'PsS4p'
	0.0386681514778681;		% yinit(22) is the initial condition for 'SWE1T'
	0.010015952644051;		% yinit(23) is the initial condition for 'SWE1P'
	0.00932949790766704;		% yinit(24) is the initial condition for 'ORI'
	0.0217532075793431;		% yinit(25) is the initial condition for 'CKIT'
	8.64531534982204E-11;		% yinit(26) is the initial condition for 'CKIP'
	0.174664621792243;		% yinit(27) is the initial condition for 'CLB2T'
	0.0644525639544721;		% yinit(28) is the initial condition for 'SPN'
	0.941791330210479;		% yinit(29) is the initial condition for 'PmS6Mp'
	0.0631248047474154;		% yinit(30) is the initial condition for 'MAD2A'
	0.520026112413009;		% yinit(31) is the initial condition for 'TEM1'
	0.599316579065192;		% yinit(32) is the initial condition for 'CDC20T'
	0.0311866692023255;		% yinit(33) is the initial condition for 'BUD'
	0.0;		% yinit(34) is the initial condition for 'CDC14c'
	0.786839982382206;		% yinit(35) is the initial condition for 'CDH1A'
	0.200718635276327;		% yinit(36) is the initial condition for 'PmS6S4p'
	0.271646225338884;		% yinit(37) is the initial condition for 'CDC5T'
];
if nargin >= 2
	if length(argYinit) > 0
		%
		% initial conditions overridden by function arguments
		%
		yinit = argYinit;
	end
end
%
% Default Parameters
%   constants are only those "Constants" from the Math Description that are just floating point numbers (no identifiers)
%   note: constants of the form "A_init" are really initial conditions and are treated in "yinit"
%
param = [
	1.0;		% param(1) is 'ks4s6p'
	4.8;		% param(2) is 'wa_polo_b2'
	0.200718635276327;		% param(3) is 'PmS6S4p_init'
	2.0;		% param(4) is 'wssa1_clb2'
	1.13738570370057;		% param(5) is 'APCP_init'
	0.14;		% param(6) is 'Jspn'
	96480.0;		% param(7) is 'mlabfix_F_'
	0.0153;		% param(8) is 'kd_ki'
	13.0;		% param(9) is 'k_cdc14_exp'
	0.216;		% param(10) is 'ks4s6'
	300.0;		% param(11) is 'Vmax'
	1.0850756320167;		% param(12) is 'MASS_init'
	0.027;		% param(13) is 'CDC55_init'
	2.0;		% param(14) is 'ks_ori_e'
	0.7;		% param(15) is 'wi_apc'
	8.0E-4;		% param(16) is 'ks_cdc20'
	2.5;		% param(17) is 'wi_tem1_bub2'
	0.0386681514778681;		% param(18) is 'SWE1T_init'
	0.6;		% param(19) is 'wi_mad2'
	0.136080572502587;		% param(20) is 'PsS6S4Wp_init'
	0.25;		% param(21) is 'kd_k2'
	1.747;		% param(22) is 'wdp_cki_14'
	0.2;		% param(23) is 'kmass_clb5'
	1.8;		% param(24) is 'kmass_clb2'
	0.1;		% param(25) is 'ks_cln2_sbf'
	0.5;		% param(26) is 'ESP1T'
	0.01;		% param(27) is 'wp_s4s6pw_b5'
	2.87;		% param(28) is 'wp_s4s6pw_b2'
	0.0156;		% param(29) is 'ks_clb5_mbf'
	0.2;		% param(30) is 'KEZ'
	0.03;		% param(31) is 'ks_pds1_mbf'
	0.03;		% param(32) is 'ks_pds1'
	0.0;		% param(33) is 'LumpedJ_reaction_49'
	0.0;		% param(34) is 'LumpedJ_reaction_48'
	0.0;		% param(35) is 'LumpedJ_reaction_47'
	1.0;		% param(36) is 'Bub2T'
	1.1;		% param(37) is 'k_mbf_swi4'
	1.0;		% param(38) is 'rho_14_net1'
	0.174664621792243;		% param(39) is 'CLB2T_init'
	0.786839982382206;		% param(40) is 'CDH1A_init'
	3.85;		% param(41) is 'wi_bub2_lo'
	0.303096279050485;		% param(42) is 'SWI5T_init'
	0.005;		% param(43) is 'ks_swi5'
	0.5;		% param(44) is 'wdp_swe1'
	2.51;		% param(45) is 'wdp_net1_14'
	2.5;		% param(46) is 'kydj1'
	1.7;		% param(47) is 'wi_mcm1'
	30.0;		% param(48) is 'wa_mad2'
	10.0;		% param(49) is 'Whi5'
	0.213;		% param(50) is 'ks4'
	0.0;		% param(51) is 'ks_cln2'
	6.6;		% param(52) is 'wp_net1_en'
	0.4;		% param(53) is 'wp_s6s4_b2'
	0.0;		% param(54) is 'PsS4p_init'
	3.55;		% param(55) is 'NET1T'
	0.269830552689792;		% param(56) is 'BCK2_init'
	2.0;		% param(57) is 'Pm_0'
	0.0380991441945013;		% param(58) is 'CLB5T_init'
	0.981;		% param(59) is 'wi_ppx_p1'
	10.0;		% param(60) is 'gammki'
	0.38;		% param(61) is 'e_bud_b5'
	0.0;		% param(62) is 'SPNALIGN_init'
	0.275187325794182;		% param(63) is 'NET1A_init'
	0.05;		% param(64) is 'Theta_cleave'
	1000.0;		% param(65) is 'K_millivolts_per_volt'
	1.0E-9;		% param(66) is 'mlabfix_K_GHK_'
	25.0;		% param(67) is 'APCT'
	0.85;		% param(68) is 'wa_cdc15_14'
	6.7;		% param(69) is 'wi_bub2_lte_lo'
	1.0;		% param(70) is 'ORIFLAG_init'
	1.0;		% param(71) is 'gamm'
	0.5;		% param(72) is 'wi_s6s4'
	2.9;		% param(73) is 'kmass_ydj1'
	0.0243221073519805;		% param(74) is 'BUB2_init'
	0.625;		% param(75) is 'wa_apc_b2'
	0.32;		% param(76) is 'kd_cdc20'
	0.08;		% param(77) is 'kd_swi5'
	0.15;		% param(78) is 'kd_clb2_20_i'
	0.504749296263001;		% param(79) is 'CDC5A_init'
	0.8;		% param(80) is 'wa_bub2_px'
	9.648E-5;		% param(81) is 'mlabfix_F_nmol_'
	0.0;		% param(82) is 'PsS6S4p_init'
	0.1;		% param(83) is 'ks_n3'
	8.5;		% param(84) is 'wp_s4s4p_k2'
	0.8;		% param(85) is 'kd_clb5_20'
	0.001;		% param(86) is 'ks_swe1'
	0.2;		% param(87) is 'kd_cln3'
	0.859443593213341;		% param(88) is 'CDC15_init'
	1.15;		% param(89) is 'wp_cki_n2'
	10.0;		% param(90) is 'wa_mcm1_b2'
	10.0;		% param(91) is 'sigcdc15'
	0.2;		% param(92) is 'wa_swi5'
	9.1;		% param(93) is 'wi_cdh1_b5'
	0.162;		% param(94) is 'wi_cdh1_b2'
	1.1;		% param(95) is 'wa_cdh1_14'
	0.19772;		% param(96) is 'ks_clb2_m1'
	0.288;		% param(97) is 'wp_net1_15'
	0.5;		% param(98) is 'e_ori_b5'
	1.7;		% param(99) is 'kd_pds_20_i'
	0.35;		% param(100) is 'e_ori_b2'
	1.05;		% param(101) is 'wp_clb2_swe1'
	0.0225;		% param(102) is 'wp_net1_b2'
	0.599316579065192;		% param(103) is 'CDC20T_init'
	300.0;		% param(104) is 'mlabfix_T_'
	0.022;		% param(105) is 'k_cdc14_imp'
	0.8;		% param(106) is 'wi_s4s6pw'
	0.06;		% param(107) is 'kd_ori'
	0.0;		% param(108) is 'UDNA_init'
	0.03;		% param(109) is 'ks_swi5_m1'
	0.13;		% param(110) is 'ks_k2'
	0.0;		% param(111) is 'PsS6S4np_init'
	0.5;		% param(112) is 'wi_s6mb'
	3.6;		% param(113) is 'wpm_s6s4_n3'
	0.28;		% param(114) is 'wpm_s6s4_n2'
	3.3;		% param(115) is 'wp_s6mb_n3'
	2.71;		% param(116) is 'wa_bub2'
	0.2;		% param(117) is 'wp_s6mb_n2'
	1.0;		% param(118) is 'hCDC6'
	0.01;		% param(119) is 'kd_swe1'
	0.0;		% param(120) is 'CLB2nd'
	0.24;		% param(121) is 'ks_spn'
	8.64531534982204E-11;		% param(122) is 'CKIP_init'
	1.5;		% param(123) is 'kd_polo_h1'
	0.7;		% param(124) is 'wdp_cki'
	8314.0;		% param(125) is 'mlabfix_R_'
	10.0;		% param(126) is 'sig'
	0.03;		% param(127) is 'kd_spn'
	0.0149;		% param(128) is 'wi_cdc15_b2'
	0.0;		% param(129) is 'ks_polo'
	0.22;		% param(130) is 'ks_polo_m1'
	1.5;		% param(131) is 'wdp_clb2'
	0.5;		% param(132) is 'kd_cdh1_swe1'
	1.0;		% param(133) is 'wdp_net1_px'
	0.0840482413806777;		% param(134) is 'CLN3_init'
	0.5;		% param(135) is 'gammcp'
	0.22;		% param(136) is 'wp_net1'
	2.0;		% param(137) is 'TEM1T'
	0.63;		% param(138) is 'ks6m'
	0.6;		% param(139) is 'wpm_s6s4_k2'
	1.0;		% param(140) is 'wi_swi5_b2'
	0.04;		% param(141) is 'wp_s6mb_k2'
	0.2;		% param(142) is 'wi_polo'
	0.7;		% param(143) is 'kd_cdh1_swe1p'
	12.0;		% param(144) is 'kmass_cln3'
	1.0;		% param(145) is 'h_sic1'
	2.5;		% param(146) is 'kd_pds_20'
	1.1;		% param(147) is 'wa_tem1_polo'
	8.43;		% param(148) is 'wp_s4s6pw_n3'
	0.010015952644051;		% param(149) is 'SWE1P_init'
	0.01;		% param(150) is 'wp_s4s6pw_n2'
	0.01;		% param(151) is 'kd_bud'
	90.0;		% param(152) is 'MDT'
	0.0311866692023255;		% param(153) is 'BUD_init'
	0.3;		% param(154) is 'kd_polo'
	0.818;		% param(155) is 'MassatGAL'
	0.23;		% param(156) is 'wi_cdc15'
	25.0;		% param(157) is 'MAD2T'
	0.16;		% param(158) is 'k_mbf_mp'
	1.3;		% param(159) is 'ks_bud_e'
	0.2;		% param(160) is 'kd_swe1p'
	0.6;		% param(161) is 'kd_clb2_h1'
	0.1;		% param(162) is 'gammtem'
	0.5;		% param(163) is 'wi_s4s4p'
	0.032;		% param(164) is 'wa_cdh1'
	8.0E-4;		% param(165) is 'ks_clb5'
	0.196030848449198;		% param(166) is 'CLN2_init'
	0.005;		% param(167) is 'ks_clb2'
	4.5;		% param(168) is 'wp_s4s4p_b2'
	0.0631248047474154;		% param(169) is 'MAD2A_init'
	2.0;		% param(170) is 'Ps_0'
	1.5;		% param(171) is 'wp_swe1_clb2'
	2.1;		% param(172) is 'wp_s4s6pw_k2'
	0.0011520415;		% param(173) is 'ks_cki'
	0.02745;		% param(174) is 'ks_cki_swi5'
	1.0;		% param(175) is 'MCM1T'
	0.199;		% param(176) is 'KEZ2'
	0.01;		% param(177) is 'kd_pds'
	1.0;		% param(178) is 'CDC55T'
	30.0;		% param(179) is 'Swi6'
	5.5;		% param(180) is 'Swi4'
	1.0;		% param(181) is 'CDC15T'
	0.3;		% param(182) is 'e_bud_n3'
	0.45;		% param(183) is 'e_bud_n2'
	0.520026112413009;		% param(184) is 'TEM1_init'
	3.141592653589793;		% param(185) is 'mlabfix_PI_'
	2.0;		% param(186) is 'CDC14T'
	0.5;		% param(187) is 'kd_kip'
	0.941791330210479;		% param(188) is 'PmS6Mp_init'
	0.035;		% param(189) is 'kd_clb5'
	0.0;		% param(190) is 'TEV'
	0.0086;		% param(191) is 'kd_clb2'
	3.0;		% param(192) is 'kmass_bck2'
	9.5;		% param(193) is 'wp_cki_b5'
	1.65;		% param(194) is 'wp_cki_b2'
	0.4586;		% param(195) is 'f'
	5.5;		% param(196) is 'Mbp1'
	6.02E11;		% param(197) is 'mlabfix_N_pmol_'
	1.142;		% param(198) is 'kd_clb2_20'
	1.0;		% param(199) is 'CDH1T'
	1.0898;		% param(200) is 'MassatGlu'
	0.2;		% param(201) is 'ks_cdc20_m1'
	0.25;		% param(202) is 'kd_clb5_20_i'
	5.1;		% param(203) is 'wa_swi5_14'
	(1.0 ./ 602.0);		% param(204) is 'KMOLE'
	0.0644525639544721;		% param(205) is 'SPN_init'
	0.0337966937259905;		% param(206) is 'PDS1T_init'
	0.9;		% param(207) is 'wi_cdh1_n2'
	1.0;		% param(208) is 'Size_cell'
	0.2;		% param(209) is 'k_clb5'
	1.0;		% param(210) is 'SSA1T'
	0.0;		% param(211) is 'CDC14c_init'
	0.0;		% param(212) is 'PsS6Mp_init'
	0.0217532075793431;		% param(213) is 'CKIT_init'
	0.271646225338884;		% param(214) is 'CDC5T_init'
	0.5;		% param(215) is 'wa_tem1'
	4.62;		% param(216) is 'wpm_s6s4_b5'
	0.035;		% param(217) is 'wpm_s6s4_b2'
	0.135;		% param(218) is 'kd_n2'
	6.2;		% param(219) is 'wp_s6mb_b5'
	0.055;		% param(220) is 'wdp_net'
	0.03;		% param(221) is 'wp_s6mb_b2'
	0.0;		% param(222) is 'HU'
	0.00932949790766704;		% param(223) is 'ORI_init'
	0.05;		% param(224) is 'wa_bub2_14'
	0.007;		% param(225) is 'ks_sbf_swe1'
	0.0;		% param(226) is 'NOC'
	1.0;		% param(227) is 'wssa1'
];
if nargin >= 3
	if length(argParam) > 0
		%
		% parameter values overridden by function arguments
		%
		param = argParam;
	end
end
%
% invoke the integrator
%
[T,Y] = ode15s(@f,timeSpan,yinit,odeset('OutputFcn',@odeplot),param,yinit);

% get the solution
all = zeros(length(T), numVars);
for i = 1:length(T)
	all(i,:) = getRow(T(i), Y(i,:), yinit, param);
end

allValues = all;
end

% -------------------------------------------------------
% get row data
function rowValue = getRow(t,y,y0,p)
	% State Variables
	CDC5A = y(1);
	CDC55 = y(2);
	UDNA = y(3);
	PsS6S4p = y(4);
	SPNALIGN = y(5);
	CDC15 = y(6);
	PDS1T = y(7);
	PsS6S4Wp = y(8);
	CLN3 = y(9);
	CLN2 = y(10);
	BUB2 = y(11);
	PsS6Mp = y(12);
	NET1A = y(13);
	PsS6S4np = y(14);
	APCP = y(15);
	MASS = y(16);
	SWI5T = y(17);
	ORIFLAG = y(18);
	BCK2 = y(19);
	CLB5T = y(20);
	PsS4p = y(21);
	SWE1T = y(22);
	SWE1P = y(23);
	ORI = y(24);
	CKIT = y(25);
	CKIP = y(26);
	CLB2T = y(27);
	SPN = y(28);
	PmS6Mp = y(29);
	MAD2A = y(30);
	TEM1 = y(31);
	CDC20T = y(32);
	BUD = y(33);
	CDC14c = y(34);
	CDH1A = y(35);
	PmS6S4p = y(36);
	CDC5T = y(37);
	% Constants
	ks4s6p = p(1);
	wa_polo_b2 = p(2);
	PmS6S4p_init = p(3);
	wssa1_clb2 = p(4);
	APCP_init = p(5);
	Jspn = p(6);
	mlabfix_F_ = p(7);
	kd_ki = p(8);
	k_cdc14_exp = p(9);
	ks4s6 = p(10);
	Vmax = p(11);
	MASS_init = p(12);
	CDC55_init = p(13);
	ks_ori_e = p(14);
	wi_apc = p(15);
	ks_cdc20 = p(16);
	wi_tem1_bub2 = p(17);
	SWE1T_init = p(18);
	wi_mad2 = p(19);
	PsS6S4Wp_init = p(20);
	kd_k2 = p(21);
	wdp_cki_14 = p(22);
	kmass_clb5 = p(23);
	kmass_clb2 = p(24);
	ks_cln2_sbf = p(25);
	ESP1T = p(26);
	wp_s4s6pw_b5 = p(27);
	wp_s4s6pw_b2 = p(28);
	ks_clb5_mbf = p(29);
	KEZ = p(30);
	ks_pds1_mbf = p(31);
	ks_pds1 = p(32);
	LumpedJ_reaction_49 = p(33);
	LumpedJ_reaction_48 = p(34);
	LumpedJ_reaction_47 = p(35);
	Bub2T = p(36);
	k_mbf_swi4 = p(37);
	rho_14_net1 = p(38);
	CLB2T_init = p(39);
	CDH1A_init = p(40);
	wi_bub2_lo = p(41);
	SWI5T_init = p(42);
	ks_swi5 = p(43);
	wdp_swe1 = p(44);
	wdp_net1_14 = p(45);
	kydj1 = p(46);
	wi_mcm1 = p(47);
	wa_mad2 = p(48);
	Whi5 = p(49);
	ks4 = p(50);
	ks_cln2 = p(51);
	wp_net1_en = p(52);
	wp_s6s4_b2 = p(53);
	PsS4p_init = p(54);
	NET1T = p(55);
	BCK2_init = p(56);
	Pm_0 = p(57);
	CLB5T_init = p(58);
	wi_ppx_p1 = p(59);
	gammki = p(60);
	e_bud_b5 = p(61);
	SPNALIGN_init = p(62);
	NET1A_init = p(63);
	Theta_cleave = p(64);
	K_millivolts_per_volt = p(65);
	mlabfix_K_GHK_ = p(66);
	APCT = p(67);
	wa_cdc15_14 = p(68);
	wi_bub2_lte_lo = p(69);
	ORIFLAG_init = p(70);
	gamm = p(71);
	wi_s6s4 = p(72);
	kmass_ydj1 = p(73);
	BUB2_init = p(74);
	wa_apc_b2 = p(75);
	kd_cdc20 = p(76);
	kd_swi5 = p(77);
	kd_clb2_20_i = p(78);
	CDC5A_init = p(79);
	wa_bub2_px = p(80);
	mlabfix_F_nmol_ = p(81);
	PsS6S4p_init = p(82);
	ks_n3 = p(83);
	wp_s4s4p_k2 = p(84);
	kd_clb5_20 = p(85);
	ks_swe1 = p(86);
	kd_cln3 = p(87);
	CDC15_init = p(88);
	wp_cki_n2 = p(89);
	wa_mcm1_b2 = p(90);
	sigcdc15 = p(91);
	wa_swi5 = p(92);
	wi_cdh1_b5 = p(93);
	wi_cdh1_b2 = p(94);
	wa_cdh1_14 = p(95);
	ks_clb2_m1 = p(96);
	wp_net1_15 = p(97);
	e_ori_b5 = p(98);
	kd_pds_20_i = p(99);
	e_ori_b2 = p(100);
	wp_clb2_swe1 = p(101);
	wp_net1_b2 = p(102);
	CDC20T_init = p(103);
	mlabfix_T_ = p(104);
	k_cdc14_imp = p(105);
	wi_s4s6pw = p(106);
	kd_ori = p(107);
	UDNA_init = p(108);
	ks_swi5_m1 = p(109);
	ks_k2 = p(110);
	PsS6S4np_init = p(111);
	wi_s6mb = p(112);
	wpm_s6s4_n3 = p(113);
	wpm_s6s4_n2 = p(114);
	wp_s6mb_n3 = p(115);
	wa_bub2 = p(116);
	wp_s6mb_n2 = p(117);
	hCDC6 = p(118);
	kd_swe1 = p(119);
	CLB2nd = p(120);
	ks_spn = p(121);
	CKIP_init = p(122);
	kd_polo_h1 = p(123);
	wdp_cki = p(124);
	mlabfix_R_ = p(125);
	sig = p(126);
	kd_spn = p(127);
	wi_cdc15_b2 = p(128);
	ks_polo = p(129);
	ks_polo_m1 = p(130);
	wdp_clb2 = p(131);
	kd_cdh1_swe1 = p(132);
	wdp_net1_px = p(133);
	CLN3_init = p(134);
	gammcp = p(135);
	wp_net1 = p(136);
	TEM1T = p(137);
	ks6m = p(138);
	wpm_s6s4_k2 = p(139);
	wi_swi5_b2 = p(140);
	wp_s6mb_k2 = p(141);
	wi_polo = p(142);
	kd_cdh1_swe1p = p(143);
	kmass_cln3 = p(144);
	h_sic1 = p(145);
	kd_pds_20 = p(146);
	wa_tem1_polo = p(147);
	wp_s4s6pw_n3 = p(148);
	SWE1P_init = p(149);
	wp_s4s6pw_n2 = p(150);
	kd_bud = p(151);
	MDT = p(152);
	BUD_init = p(153);
	kd_polo = p(154);
	MassatGAL = p(155);
	wi_cdc15 = p(156);
	MAD2T = p(157);
	k_mbf_mp = p(158);
	ks_bud_e = p(159);
	kd_swe1p = p(160);
	kd_clb2_h1 = p(161);
	gammtem = p(162);
	wi_s4s4p = p(163);
	wa_cdh1 = p(164);
	ks_clb5 = p(165);
	CLN2_init = p(166);
	ks_clb2 = p(167);
	wp_s4s4p_b2 = p(168);
	MAD2A_init = p(169);
	Ps_0 = p(170);
	wp_swe1_clb2 = p(171);
	wp_s4s6pw_k2 = p(172);
	ks_cki = p(173);
	ks_cki_swi5 = p(174);
	MCM1T = p(175);
	KEZ2 = p(176);
	kd_pds = p(177);
	CDC55T = p(178);
	Swi6 = p(179);
	Swi4 = p(180);
	CDC15T = p(181);
	e_bud_n3 = p(182);
	e_bud_n2 = p(183);
	TEM1_init = p(184);
	mlabfix_PI_ = p(185);
	CDC14T = p(186);
	kd_kip = p(187);
	PmS6Mp_init = p(188);
	kd_clb5 = p(189);
	TEV = p(190);
	kd_clb2 = p(191);
	kmass_bck2 = p(192);
	wp_cki_b5 = p(193);
	wp_cki_b2 = p(194);
	f = p(195);
	Mbp1 = p(196);
	mlabfix_N_pmol_ = p(197);
	kd_clb2_20 = p(198);
	CDH1T = p(199);
	MassatGlu = p(200);
	ks_cdc20_m1 = p(201);
	kd_clb5_20_i = p(202);
	wa_swi5_14 = p(203);
	KMOLE = p(204);
	SPN_init = p(205);
	PDS1T_init = p(206);
	wi_cdh1_n2 = p(207);
	Size_cell = p(208);
	k_clb5 = p(209);
	SSA1T = p(210);
	CDC14c_init = p(211);
	PsS6Mp_init = p(212);
	CKIT_init = p(213);
	CDC5T_init = p(214);
	wa_tem1 = p(215);
	wpm_s6s4_b5 = p(216);
	wpm_s6s4_b2 = p(217);
	kd_n2 = p(218);
	wp_s6mb_b5 = p(219);
	wdp_net = p(220);
	wp_s6mb_b2 = p(221);
	HU = p(222);
	ORI_init = p(223);
	wa_bub2_14 = p(224);
	ks_sbf_swe1 = p(225);
	NOC = p(226);
	wssa1 = p(227);
	% Functions
	Wclb2 = ((wp_clb2_swe1 .* SWE1T) - wdp_clb2);
	CLB2f = (((CLB2T + CLB2nd) .* ((1.0 - ((CKIT .* hCDC6) ./ (CLB2T + CLB2nd + (h_sic1 .* CLB5T)))) .* (0.0 < (1.0 - ((CKIT .* hCDC6) ./ (CLB2T + CLB2nd + (h_sic1 .* CLB5T))))))) .* ((CLB5T + CLB2T + CLB2nd) > 1.0E-8));
	CLB2 = (CLB2f - (CLB2f ./ (1.0 + exp(( - sig .* Wclb2)))));
	Wswe1 = ((wp_swe1_clb2 .* CLB2) - wdp_swe1);
	ESP1 = ((ESP1T - PDS1T) .* (0.0 < (ESP1T - PDS1T)));
	GLUE = (ESP1 + TEV);
	I_SPOC = ((1.0 - NOC) .* ((1.0 .* ((GLUE - Theta_cleave) >= 0.0)) + (0.0 .* ~(((GLUE - Theta_cleave) >= 0.0)))));
	S6S4 = ((((Swi6 .* Swi4) ./ (Swi4 + Mbp1)) .* (Swi4 >= ((Swi6 .* Swi4) ./ (Swi4 + Mbp1)))) + (Swi4 .* ~((Swi4 >= ((Swi6 .* Swi4) ./ (Swi4 + Mbp1))))));
	PsS6S4W = ((((Whi5 .* (S6S4 >= Whi5)) + (S6S4 .* ~((S6S4 >= Whi5)))) .* (((S6S4 .* (Ps_0 >= S6S4)) + (Ps_0 .* ~((Ps_0 >= S6S4)))) >= ((Whi5 .* (S6S4 >= Whi5)) + (S6S4 .* ~((S6S4 >= Whi5)))))) + (((S6S4 .* (Ps_0 >= S6S4)) + (Ps_0 .* ~((Ps_0 >= S6S4)))) .* ~((((S6S4 .* (Ps_0 >= S6S4)) + (Ps_0 .* ~((Ps_0 >= S6S4)))) >= ((Whi5 .* (S6S4 >= Whi5)) + (S6S4 .* ~((S6S4 >= Whi5))))))));
	PsS6S4free = (((S6S4 .* (Ps_0 >= S6S4)) + (Ps_0 .* ~((Ps_0 >= S6S4)))) - PsS6S4W);
	PsS6S4 = ((0.0 .* (0.0 >= PsS6S4free)) + (PsS6S4free .* ~((0.0 >= PsS6S4free))));
	S6S4free = ((S6S4 - PsS6S4W) - PsS6S4);
	PmS6S4 = ((S6S4free .* ((0.15 .* Pm_0) >= S6S4free)) + ((0.15 .* Pm_0) .* ~(((0.15 .* Pm_0) >= S6S4free))));
	S4free = (((Swi4 - PsS6S4) - PsS6S4W) - PmS6S4);
	LumpedJ_reaction_53 = ((kd_swe1p + (kd_cdh1_swe1p .* CDH1A)) .* SWE1P);
	LumpedJ_reaction_52 = (gamm .* ((SWE1T ./ (1.0 + exp(( - sig .* Wswe1)))) - SWE1P));
	LumpedJ_reaction_51 = (((kd_swe1 + (kd_cdh1_swe1 .* CDH1A)) .* (SWE1T - SWE1P)) + ((kd_swe1p + (kd_cdh1_swe1p .* CDH1A)) .* SWE1P));
	VS2 = (PsS6S4Wp + (ks4s6p .* PsS6S4p) + (ks4s6 .* PsS6S4np) + (ks6m .* PsS6Mp) + (ks4 .* PsS4p));
	LumpedJ_reaction_50 = (ks_swe1 + (ks_sbf_swe1 .* VS2));
	S6M = ((((Swi6 .* Mbp1) ./ (Swi4 + Mbp1)) .* (Mbp1 >= ((Swi6 .* Mbp1) ./ (Swi4 + Mbp1)))) + (Mbp1 .* ~((Mbp1 >= ((Swi6 .* Mbp1) ./ (Swi4 + Mbp1))))));
	PmS6M = ((S6M .* ((0.85 .* Pm_0) >= S6M)) + ((0.85 .* Pm_0) .* ~(((0.85 .* Pm_0) >= S6M))));
	S6Mfree = (S6M - PmS6M);
	Wmcm1 = ((wa_mcm1_b2 .* CLB2) - wi_mcm1);
	MCM1 = (MCM1T ./ (1.0 + exp(( - sig .* Wmcm1))));
	CDC14 = ((0.0 .* (0.0 >= (CDC14T - (rho_14_net1 .* NET1A)))) + ((CDC14T - (rho_14_net1 .* NET1A)) .* ~((0.0 >= (CDC14T - (rho_14_net1 .* NET1A))))));
	Wbub2 = ((((wa_bub2_px .* CDC55) + (wa_bub2_14 .* CDC14) + wa_bub2) - (wi_bub2_lte_lo .* CDC5A .* I_SPOC)) - (wi_bub2_lo .* CDC5A));
	LumpedJ_reaction_46 = (gamm .* ((Bub2T ./ (1.0 + exp(( - sig .* Wbub2)))) - BUB2));
	Wpolo = ((wa_polo_b2 .* CLB2) - wi_polo);
	LumpedJ_reaction_45 = (gamm .* ((CDC5T ./ (1.0 + exp(( - sig .* Wpolo)))) - CDC5A));
	LumpedJ_reaction_44 = ((kd_polo + (kd_polo_h1 .* CDH1A)) .* CDC5T);
	LumpedJ_reaction_43 = (ks_polo + (ks_polo_m1 .* MCM1));
	Wtem1 = (((wa_tem1_polo .* CDC5A) + wa_tem1) - (wi_tem1_bub2 .* BUB2));
	LumpedJ_reaction_42 = (gammtem .* ((TEM1T ./ (1.0 + exp(( - sig .* Wtem1)))) - TEM1));
	Wcdc15 = (((wa_cdc15_14 .* CDC14) - wi_cdc15) - (wi_cdc15_b2 .* CLB2));
	LumpedJ_reaction_41 = (gamm .* ((CDC15T ./ (1.0 + exp(( - sigcdc15 .* Wcdc15)))) - CDC15));
	CDC15f = (((CDC15 - TEM1) .* ((CDC15 - TEM1) >= 0.0)) + (0.0 .* ~(((CDC15 - TEM1) >= 0.0))));
	MEN = (((1.0 .* (CDC5A >= 0.0)) + (0.0 .* ~((CDC5A >= 0.0)))) .* ((CDC15 .* (TEM1 >= CDC15)) + (TEM1 .* ~((TEM1 >= CDC15)))));
	CDC14n = (CDC14 - CDC14c);
	Wnet1 = (((wdp_net1_px .* CDC55) + wdp_net + (wdp_net1_14 .* CDC14n)) - (CDC5A .* (wp_net1 + (wp_net1_b2 .* CLB2) + (wp_net1_en .* MEN) + (wp_net1_15 .* CDC15f))));
	LumpedJ_reaction_40 = (gamm .* ((NET1T ./ (1.0 + exp(( - sig .* Wnet1)))) - NET1A));
	Psfree = ((0.0 .* (0.0 >= ((Ps_0 - PsS6S4) - PsS6S4W))) + (((Ps_0 - PsS6S4) - PsS6S4W) .* ~((0.0 >= ((Ps_0 - PsS6S4) - PsS6S4W)))));
	LumpedJ_reaction_39 = (k_cdc14_imp .* CDC14c);
	LumpedJ_reaction_38 = (k_cdc14_exp .* (CDC14 - CDC14c) .* MEN);
	Wcdc55 = ( - wi_ppx_p1 .* ESP1);
	LumpedJ_reaction_37 = (gamm .* ((CDC55T ./ (1.0 + exp(( - sig .* Wcdc55)))) - CDC55));
	CDC20A = ((0.0 .* (0.0 >= (CDC20T - MAD2A))) + ((CDC20T - MAD2A) .* ~((0.0 >= (CDC20T - MAD2A)))));
	CDC20A_APCP = ((APCP .* (CDC20A >= APCP)) + (CDC20A .* ~((CDC20A >= APCP))));
	CDC20A_APC = (((APCT - APCP) .* ((CDC20A - CDC20A_APCP) >= (APCT - APCP))) + ((CDC20A - CDC20A_APCP) .* ~(((CDC20A - CDC20A_APCP) >= (APCT - APCP)))));
	LumpedJ_reaction_36 = ((kd_pds + (kd_pds_20 .* CDC20A_APCP) + (kd_pds_20_i .* CDC20A_APC)) .* PDS1T);
	Pmfree = ((0.0 .* (0.0 >= ((Pm_0 - PmS6S4) - PmS6M))) + (((Pm_0 - PmS6S4) - PmS6M) .* ~((0.0 >= ((Pm_0 - PmS6S4) - PmS6M)))));
	VS5 = ((k_mbf_mp .* PmS6Mp) + (k_mbf_swi4 .* PmS6S4p) + ((Pmfree .* k_clb5 .* ks_clb5) ./ (ks_clb5_mbf + 1.0E-8)));
	LumpedJ_reaction_35 = (ks_pds1 + (ks_pds1_mbf .* VS5));
	CLB5 = ((CLB5T .* ((1.0 - ((CKIT .* h_sic1) ./ (CLB2T + CLB2nd + CLB5T))) .* (0.0 < (1.0 - ((CKIT .* h_sic1) ./ (CLB2T + CLB2nd + CLB5T)))))) .* ((CLB5T + CLB2T + CLB2nd) > 1.0E-8));
	Wcdh1 = ((wa_cdh1 + (wa_cdh1_14 .* CDC14)) - ((wi_cdh1_n2 .* CLN2) + (wi_cdh1_b5 .* CLB5) + (wi_cdh1_b2 .* CLB2)));
	SAC = (((UDNA .* (1.0 - SPNALIGN)) .* ((UDNA .* (1.0 - SPNALIGN)) >= HU)) + (HU .* ~(((UDNA .* (1.0 - SPNALIGN)) >= HU))));
	Wmad2 = ((wa_mad2 .* SAC) - wi_mad2);
	LumpedJ_reaction_34 = (gamm .* ((MAD2T ./ (1.0 + exp(( - sig .* Wmad2)))) - MAD2A));
	LumpedJ_reaction_33 = (kd_cdc20 .* CDC20T);
	LumpedJ_reaction_32 = (ks_cdc20 + (ks_cdc20_m1 .* MCM1));
	Wapc = ((wa_apc_b2 .* CLB2) - wi_apc);
	LumpedJ_reaction_31 = (gammcp .* ((APCT ./ (1.0 + exp(( - sig .* Wapc)))) - APCP));
	LumpedJ_reaction_30 = (gamm .* ((CDH1T ./ (1.0 + exp(( - sig .* Wcdh1)))) - CDH1A));
	Wssa1 = ((wssa1_clb2 .* CLB2) - wssa1);
	LumpedJ_reaction_29 = (kd_swi5 .* SWI5T);
	LumpedJ_reaction_28 = (ks_swi5 + (ks_swi5_m1 .* MCM1));
	LumpedJ_reaction_27 = (kd_spn .* SPN);
	LumpedJ_reaction_26 = (ks_spn .* ((1.0 .* ((CLB2 - Jspn) >= 0.0)) + (0.0 .* ~(((CLB2 - Jspn) >= 0.0)))));
	LumpedJ_reaction_25 = (kd_ori .* ORI);
	PsS4 = ((S4free .* (Psfree >= S4free)) + (Psfree .* ~((Psfree >= S4free))));
	LumpedJ_reaction_24 = (ks_ori_e .* ((e_ori_b5 .* CLB5) + (e_ori_b2 .* CLB2)));
	LumpedJ_reaction_23 = (kd_bud .* BUD);
	SSA1p = (SSA1T ./ (1.0 + exp(( - sig .* Wssa1))));
	Ydj1 = (kydj1 .* (1.0 - exp( - (MASS ./ kmass_ydj1))));
	SSA1free = ((((SSA1T - SSA1p) - Ydj1) .* (((SSA1T - SSA1p) - Ydj1) >= 0.0)) + (0.0 .* ~((((SSA1T - SSA1p) - Ydj1) >= 0.0))));
	CLN3n = (((CLN3 - (SSA1free + SSA1p)) .* ((CLN3 - (SSA1free + SSA1p)) >= 0.0)) + (0.0 .* ~(((CLN3 - (SSA1free + SSA1p)) >= 0.0))));
	LumpedJ_reaction_22 = (ks_bud_e .* ((e_bud_n3 .* CLN3n) + (e_bud_n2 .* CLN2) + (e_bud_b5 .* CLB5)));
	LumpedJ_reaction_21 = ((kd_clb2 + (kd_clb2_20 .* CDC20A_APCP) + (kd_clb2_20_i .* CDC20A_APC) + (kd_clb2_h1 .* CDH1A)) .* CLB2T);
	LumpedJ_reaction_20 = ((ks_clb2 + (ks_clb2_m1 .* MCM1)) .* (1.0 - exp( - (MASS ./ kmass_clb2))));
	SPNvsESP = (((1.0 .* ((ks_pds1 .* ks_pds1_mbf) >= 0.0)) + (0.0 .* ~(((ks_pds1 .* ks_pds1_mbf) >= 0.0)))) .* ((1.0 .* ((ESP1 - Theta_cleave) >= 0.0)) + (0.0 .* ~(((ESP1 - Theta_cleave) >= 0.0)))) .* ((1.0 .* ((1.0 - SPN) >= 0.0)) + (0.0 .* ~(((1.0 - SPN) >= 0.0)))) .* ((1.0 .* ((CLB2 - KEZ2) >= 0.0)) + (0.0 .* ~(((CLB2 - KEZ2) >= 0.0)))));
	Ws6s4w = ((((wp_s4s6pw_n3 .* CLN3n) + (wp_s4s6pw_k2 .* BCK2) + (wp_s4s6pw_n2 .* CLN2) + (wp_s4s6pw_b5 .* CLB5)) - wi_s4s6pw) - (CLB2 .* wp_s4s6pw_b2));
	LumpedJ_reaction_19 = ((kd_clb5 + (kd_clb5_20 .* CDC20A_APCP) + (kd_clb5_20_i .* CDC20A_APC)) .* CLB5T);
	LumpedJ_reaction_18 = ((ks_clb5 + (ks_clb5_mbf .* VS5)) .* (1.0 - exp( - (MASS ./ kmass_clb5))));
	LumpedJ_reaction_17 = (kd_kip .* CKIP);
	Wcki = ((((wp_cki_n2 .* CLN2) + (wp_cki_b5 .* CLB5) + (wp_cki_b2 .* CLB2)) - wdp_cki) - (wdp_cki_14 .* CDC14));
	LumpedJ_reaction_16 = (gammki .* ((CKIT ./ (1.0 + exp(( - sig .* Wcki)))) - CKIP));
	LumpedJ_reaction_15 = ((kd_ki .* (CKIT - CKIP)) + (kd_kip .* CKIP));
	Wswi5 = (( - wi_swi5_b2 .* CLB2) + wa_swi5 + (wa_swi5_14 .* CDC14));
	SWI5A = (SWI5T ./ (1.0 + exp(( - sig .* Wswi5))));
	LumpedJ_reaction_14 = (ks_cki + (ks_cki_swi5 .* SWI5A));
	LumpedJ_reaction_13 = (kd_n2 .* CLN2);
	LumpedJ_reaction_12 = (ks_cln2 + (ks_cln2_sbf .* VS2));
	Ws4 = (((wp_s4s4p_k2 .* BCK2) - wi_s4s4p) - (CLB2 .* wp_s4s4p_b2));
	LumpedJ_reaction_11 = (gamm .* ((PsS4 ./ (1.0 + exp(( - sig .* Ws4)))) - PsS4p));
	Ws6m = ((((wp_s6mb_n3 .* CLN3n) + (wp_s6mb_k2 .* BCK2) + (wp_s6mb_n2 .* CLN2) + (wp_s6mb_b5 .* CLB5)) - wi_s6mb) - (CLB2 .* wp_s6mb_b2));
	Psfree2 = ((0.0 .* (0.0 >= (((Ps_0 - PsS6S4) - PsS6S4W) - PsS4))) + ((((Ps_0 - PsS6S4) - PsS6S4W) - PsS4) .* ~((0.0 >= (((Ps_0 - PsS6S4) - PsS6S4W) - PsS4)))));
	PsS6M = ((S6Mfree .* (Psfree2 >= S6Mfree)) + (Psfree2 .* ~((Psfree2 >= S6Mfree))));
	LumpedJ_reaction_10 = (gamm .* ((PsS6M ./ (1.0 + exp(( - sig .* Ws6m)))) - PsS6Mp));
	Ws6s4np = ( - (((wpm_s6s4_n3 .* CLN3n) + (wpm_s6s4_k2 .* BCK2) + (wpm_s6s4_n2 .* CLN2) + (wpm_s6s4_b5 .* CLB5)) - wi_s6s4) - (CLB2 .* wp_s6s4_b2));
	Whi5free = (Whi5 - PsS6S4W);
	mu = (log(2.0) ./ MDT);
	UDNA1 = ((1.0 .* ((UDNA + HU) >= 0.0)) + (0.0 .* ~(((UDNA + HU) >= 0.0))));
	PmS4 = ((S4free .* ((0.15 .* Pmfree) >= S4free)) + ((0.15 .* Pmfree) .* ~(((0.15 .* Pmfree) >= S4free))));
	LumpedJ_reaction_9 = (gamm .* ((PsS6S4W ./ (1.0 + exp(( - sig .* Ws6s4w)))) - PsS6S4Wp));
	LumpedJ_reaction_8 = (gamm .* ((PsS6S4 ./ (1.0 + exp(( - sig .* Ws6s4np)))) - PsS6S4np));
	Ws6s4 = ((((wpm_s6s4_n3 .* CLN3n) + (wpm_s6s4_k2 .* BCK2) + (wpm_s6s4_n2 .* CLN2) + (wpm_s6s4_b5 .* CLB5)) - wi_s6s4) - (CLB2 .* wpm_s6s4_b2));
	LumpedJ_reaction_7 = (gamm .* ((PsS6S4 ./ (1.0 + exp(( - sig .* Ws6s4)))) - PsS6S4p));
	LumpedJ_reaction_6 = (gamm .* ((PmS6S4 ./ (1.0 + exp(( - sig .* Ws6s4)))) - PmS6S4p));
	LumpedJ_reaction_5 = (gamm .* ((PmS6M ./ (1.0 + exp(( - sig .* Ws6m)))) - PmS6Mp));
	LumpedJ_reaction_4 = (kd_k2 .* BCK2);
	LumpedJ_reaction_3 = (ks_k2 .* (1.0 - exp( - (MASS ./ kmass_bck2))));
	LumpedJ_reaction_2 = (kd_cln3 .* CLN3);
	LumpedJ_reaction_1 = (ks_n3 .* (1.0 - exp( - (MASS ./ kmass_cln3))));
	LumpedJ_reaction_0 = (mu .* MASS .* (1.0 - (MASS ./ Vmax)));
	% OutputFunctions

	rowValue = [CDC5A CDC55 UDNA PsS6S4p SPNALIGN CDC15 PDS1T PsS6S4Wp CLN3 CLN2 BUB2 PsS6Mp NET1A PsS6S4np APCP MASS SWI5T ORIFLAG BCK2 CLB5T PsS4p SWE1T SWE1P ORI CKIT CKIP CLB2T SPN PmS6Mp MAD2A TEM1 CDC20T BUD CDC14c CDH1A PmS6S4p CDC5T Wclb2 CLB2f CLB2 Wswe1 ESP1 GLUE I_SPOC S6S4 PsS6S4W PsS6S4free PsS6S4 S6S4free PmS6S4 S4free LumpedJ_reaction_53 LumpedJ_reaction_52 LumpedJ_reaction_51 VS2 LumpedJ_reaction_50 S6M PmS6M S6Mfree Wmcm1 MCM1 CDC14 Wbub2 LumpedJ_reaction_46 Wpolo LumpedJ_reaction_45 LumpedJ_reaction_44 LumpedJ_reaction_43 Wtem1 LumpedJ_reaction_42 Wcdc15 LumpedJ_reaction_41 CDC15f MEN CDC14n Wnet1 LumpedJ_reaction_40 Psfree LumpedJ_reaction_39 LumpedJ_reaction_38 Wcdc55 LumpedJ_reaction_37 CDC20A CDC20A_APCP CDC20A_APC LumpedJ_reaction_36 Pmfree VS5 LumpedJ_reaction_35 CLB5 Wcdh1 SAC Wmad2 LumpedJ_reaction_34 LumpedJ_reaction_33 LumpedJ_reaction_32 Wapc LumpedJ_reaction_31 LumpedJ_reaction_30 Wssa1 LumpedJ_reaction_29 LumpedJ_reaction_28 LumpedJ_reaction_27 LumpedJ_reaction_26 LumpedJ_reaction_25 PsS4 LumpedJ_reaction_24 LumpedJ_reaction_23 SSA1p Ydj1 SSA1free CLN3n LumpedJ_reaction_22 LumpedJ_reaction_21 LumpedJ_reaction_20 SPNvsESP Ws6s4w LumpedJ_reaction_19 LumpedJ_reaction_18 LumpedJ_reaction_17 Wcki LumpedJ_reaction_16 LumpedJ_reaction_15 Wswi5 SWI5A LumpedJ_reaction_14 LumpedJ_reaction_13 LumpedJ_reaction_12 Ws4 LumpedJ_reaction_11 Ws6m Psfree2 PsS6M LumpedJ_reaction_10 Ws6s4np Whi5free mu UDNA1 PmS4 LumpedJ_reaction_9 LumpedJ_reaction_8 Ws6s4 LumpedJ_reaction_7 LumpedJ_reaction_6 LumpedJ_reaction_5 LumpedJ_reaction_4 LumpedJ_reaction_3 LumpedJ_reaction_2 LumpedJ_reaction_1 LumpedJ_reaction_0 ];
end

% -------------------------------------------------------
% ode rate
function dydt = f(t,y,p,y0)
	% State Variables
	CDC5A = y(1);
	CDC55 = y(2);
	UDNA = y(3);
	PsS6S4p = y(4);
	SPNALIGN = y(5);
	CDC15 = y(6);
	PDS1T = y(7);
	PsS6S4Wp = y(8);
	CLN3 = y(9);
	CLN2 = y(10);
	BUB2 = y(11);
	PsS6Mp = y(12);
	NET1A = y(13);
	PsS6S4np = y(14);
	APCP = y(15);
	MASS = y(16);
	SWI5T = y(17);
	ORIFLAG = y(18);
	BCK2 = y(19);
	CLB5T = y(20);
	PsS4p = y(21);
	SWE1T = y(22);
	SWE1P = y(23);
	ORI = y(24);
	CKIT = y(25);
	CKIP = y(26);
	CLB2T = y(27);
	SPN = y(28);
	PmS6Mp = y(29);
	MAD2A = y(30);
	TEM1 = y(31);
	CDC20T = y(32);
	BUD = y(33);
	CDC14c = y(34);
	CDH1A = y(35);
	PmS6S4p = y(36);
	CDC5T = y(37);
	% Constants
	ks4s6p = p(1);
	wa_polo_b2 = p(2);
	PmS6S4p_init = p(3);
	wssa1_clb2 = p(4);
	APCP_init = p(5);
	Jspn = p(6);
	mlabfix_F_ = p(7);
	kd_ki = p(8);
	k_cdc14_exp = p(9);
	ks4s6 = p(10);
	Vmax = p(11);
	MASS_init = p(12);
	CDC55_init = p(13);
	ks_ori_e = p(14);
	wi_apc = p(15);
	ks_cdc20 = p(16);
	wi_tem1_bub2 = p(17);
	SWE1T_init = p(18);
	wi_mad2 = p(19);
	PsS6S4Wp_init = p(20);
	kd_k2 = p(21);
	wdp_cki_14 = p(22);
	kmass_clb5 = p(23);
	kmass_clb2 = p(24);
	ks_cln2_sbf = p(25);
	ESP1T = p(26);
	wp_s4s6pw_b5 = p(27);
	wp_s4s6pw_b2 = p(28);
	ks_clb5_mbf = p(29);
	KEZ = p(30);
	ks_pds1_mbf = p(31);
	ks_pds1 = p(32);
	LumpedJ_reaction_49 = p(33);
	LumpedJ_reaction_48 = p(34);
	LumpedJ_reaction_47 = p(35);
	Bub2T = p(36);
	k_mbf_swi4 = p(37);
	rho_14_net1 = p(38);
	CLB2T_init = p(39);
	CDH1A_init = p(40);
	wi_bub2_lo = p(41);
	SWI5T_init = p(42);
	ks_swi5 = p(43);
	wdp_swe1 = p(44);
	wdp_net1_14 = p(45);
	kydj1 = p(46);
	wi_mcm1 = p(47);
	wa_mad2 = p(48);
	Whi5 = p(49);
	ks4 = p(50);
	ks_cln2 = p(51);
	wp_net1_en = p(52);
	wp_s6s4_b2 = p(53);
	PsS4p_init = p(54);
	NET1T = p(55);
	BCK2_init = p(56);
	Pm_0 = p(57);
	CLB5T_init = p(58);
	wi_ppx_p1 = p(59);
	gammki = p(60);
	e_bud_b5 = p(61);
	SPNALIGN_init = p(62);
	NET1A_init = p(63);
	Theta_cleave = p(64);
	K_millivolts_per_volt = p(65);
	mlabfix_K_GHK_ = p(66);
	APCT = p(67);
	wa_cdc15_14 = p(68);
	wi_bub2_lte_lo = p(69);
	ORIFLAG_init = p(70);
	gamm = p(71);
	wi_s6s4 = p(72);
	kmass_ydj1 = p(73);
	BUB2_init = p(74);
	wa_apc_b2 = p(75);
	kd_cdc20 = p(76);
	kd_swi5 = p(77);
	kd_clb2_20_i = p(78);
	CDC5A_init = p(79);
	wa_bub2_px = p(80);
	mlabfix_F_nmol_ = p(81);
	PsS6S4p_init = p(82);
	ks_n3 = p(83);
	wp_s4s4p_k2 = p(84);
	kd_clb5_20 = p(85);
	ks_swe1 = p(86);
	kd_cln3 = p(87);
	CDC15_init = p(88);
	wp_cki_n2 = p(89);
	wa_mcm1_b2 = p(90);
	sigcdc15 = p(91);
	wa_swi5 = p(92);
	wi_cdh1_b5 = p(93);
	wi_cdh1_b2 = p(94);
	wa_cdh1_14 = p(95);
	ks_clb2_m1 = p(96);
	wp_net1_15 = p(97);
	e_ori_b5 = p(98);
	kd_pds_20_i = p(99);
	e_ori_b2 = p(100);
	wp_clb2_swe1 = p(101);
	wp_net1_b2 = p(102);
	CDC20T_init = p(103);
	mlabfix_T_ = p(104);
	k_cdc14_imp = p(105);
	wi_s4s6pw = p(106);
	kd_ori = p(107);
	UDNA_init = p(108);
	ks_swi5_m1 = p(109);
	ks_k2 = p(110);
	PsS6S4np_init = p(111);
	wi_s6mb = p(112);
	wpm_s6s4_n3 = p(113);
	wpm_s6s4_n2 = p(114);
	wp_s6mb_n3 = p(115);
	wa_bub2 = p(116);
	wp_s6mb_n2 = p(117);
	hCDC6 = p(118);
	kd_swe1 = p(119);
	CLB2nd = p(120);
	ks_spn = p(121);
	CKIP_init = p(122);
	kd_polo_h1 = p(123);
	wdp_cki = p(124);
	mlabfix_R_ = p(125);
	sig = p(126);
	kd_spn = p(127);
	wi_cdc15_b2 = p(128);
	ks_polo = p(129);
	ks_polo_m1 = p(130);
	wdp_clb2 = p(131);
	kd_cdh1_swe1 = p(132);
	wdp_net1_px = p(133);
	CLN3_init = p(134);
	gammcp = p(135);
	wp_net1 = p(136);
	TEM1T = p(137);
	ks6m = p(138);
	wpm_s6s4_k2 = p(139);
	wi_swi5_b2 = p(140);
	wp_s6mb_k2 = p(141);
	wi_polo = p(142);
	kd_cdh1_swe1p = p(143);
	kmass_cln3 = p(144);
	h_sic1 = p(145);
	kd_pds_20 = p(146);
	wa_tem1_polo = p(147);
	wp_s4s6pw_n3 = p(148);
	SWE1P_init = p(149);
	wp_s4s6pw_n2 = p(150);
	kd_bud = p(151);
	MDT = p(152);
	BUD_init = p(153);
	kd_polo = p(154);
	MassatGAL = p(155);
	wi_cdc15 = p(156);
	MAD2T = p(157);
	k_mbf_mp = p(158);
	ks_bud_e = p(159);
	kd_swe1p = p(160);
	kd_clb2_h1 = p(161);
	gammtem = p(162);
	wi_s4s4p = p(163);
	wa_cdh1 = p(164);
	ks_clb5 = p(165);
	CLN2_init = p(166);
	ks_clb2 = p(167);
	wp_s4s4p_b2 = p(168);
	MAD2A_init = p(169);
	Ps_0 = p(170);
	wp_swe1_clb2 = p(171);
	wp_s4s6pw_k2 = p(172);
	ks_cki = p(173);
	ks_cki_swi5 = p(174);
	MCM1T = p(175);
	KEZ2 = p(176);
	kd_pds = p(177);
	CDC55T = p(178);
	Swi6 = p(179);
	Swi4 = p(180);
	CDC15T = p(181);
	e_bud_n3 = p(182);
	e_bud_n2 = p(183);
	TEM1_init = p(184);
	mlabfix_PI_ = p(185);
	CDC14T = p(186);
	kd_kip = p(187);
	PmS6Mp_init = p(188);
	kd_clb5 = p(189);
	TEV = p(190);
	kd_clb2 = p(191);
	kmass_bck2 = p(192);
	wp_cki_b5 = p(193);
	wp_cki_b2 = p(194);
	f = p(195);
	Mbp1 = p(196);
	mlabfix_N_pmol_ = p(197);
	kd_clb2_20 = p(198);
	CDH1T = p(199);
	MassatGlu = p(200);
	ks_cdc20_m1 = p(201);
	kd_clb5_20_i = p(202);
	wa_swi5_14 = p(203);
	KMOLE = p(204);
	SPN_init = p(205);
	PDS1T_init = p(206);
	wi_cdh1_n2 = p(207);
	Size_cell = p(208);
	k_clb5 = p(209);
	SSA1T = p(210);
	CDC14c_init = p(211);
	PsS6Mp_init = p(212);
	CKIT_init = p(213);
	CDC5T_init = p(214);
	wa_tem1 = p(215);
	wpm_s6s4_b5 = p(216);
	wpm_s6s4_b2 = p(217);
	kd_n2 = p(218);
	wp_s6mb_b5 = p(219);
	wdp_net = p(220);
	wp_s6mb_b2 = p(221);
	HU = p(222);
	ORI_init = p(223);
	wa_bub2_14 = p(224);
	ks_sbf_swe1 = p(225);
	NOC = p(226);
	wssa1 = p(227);
	% Functions
	Wclb2 = ((wp_clb2_swe1 .* SWE1T) - wdp_clb2);
	CLB2f = (((CLB2T + CLB2nd) .* ((1.0 - ((CKIT .* hCDC6) ./ (CLB2T + CLB2nd + (h_sic1 .* CLB5T)))) .* (0.0 < (1.0 - ((CKIT .* hCDC6) ./ (CLB2T + CLB2nd + (h_sic1 .* CLB5T))))))) .* ((CLB5T + CLB2T + CLB2nd) > 1.0E-8));
	CLB2 = (CLB2f - (CLB2f ./ (1.0 + exp(( - sig .* Wclb2)))));
	Wswe1 = ((wp_swe1_clb2 .* CLB2) - wdp_swe1);
	ESP1 = ((ESP1T - PDS1T) .* (0.0 < (ESP1T - PDS1T)));
	GLUE = (ESP1 + TEV);
	I_SPOC = ((1.0 - NOC) .* ((1.0 .* ((GLUE - Theta_cleave) >= 0.0)) + (0.0 .* ~(((GLUE - Theta_cleave) >= 0.0)))));
	S6S4 = ((((Swi6 .* Swi4) ./ (Swi4 + Mbp1)) .* (Swi4 >= ((Swi6 .* Swi4) ./ (Swi4 + Mbp1)))) + (Swi4 .* ~((Swi4 >= ((Swi6 .* Swi4) ./ (Swi4 + Mbp1))))));
	PsS6S4W = ((((Whi5 .* (S6S4 >= Whi5)) + (S6S4 .* ~((S6S4 >= Whi5)))) .* (((S6S4 .* (Ps_0 >= S6S4)) + (Ps_0 .* ~((Ps_0 >= S6S4)))) >= ((Whi5 .* (S6S4 >= Whi5)) + (S6S4 .* ~((S6S4 >= Whi5)))))) + (((S6S4 .* (Ps_0 >= S6S4)) + (Ps_0 .* ~((Ps_0 >= S6S4)))) .* ~((((S6S4 .* (Ps_0 >= S6S4)) + (Ps_0 .* ~((Ps_0 >= S6S4)))) >= ((Whi5 .* (S6S4 >= Whi5)) + (S6S4 .* ~((S6S4 >= Whi5))))))));
	PsS6S4free = (((S6S4 .* (Ps_0 >= S6S4)) + (Ps_0 .* ~((Ps_0 >= S6S4)))) - PsS6S4W);
	PsS6S4 = ((0.0 .* (0.0 >= PsS6S4free)) + (PsS6S4free .* ~((0.0 >= PsS6S4free))));
	S6S4free = ((S6S4 - PsS6S4W) - PsS6S4);
	PmS6S4 = ((S6S4free .* ((0.15 .* Pm_0) >= S6S4free)) + ((0.15 .* Pm_0) .* ~(((0.15 .* Pm_0) >= S6S4free))));
	S4free = (((Swi4 - PsS6S4) - PsS6S4W) - PmS6S4);
	LumpedJ_reaction_53 = ((kd_swe1p + (kd_cdh1_swe1p .* CDH1A)) .* SWE1P);
	LumpedJ_reaction_52 = (gamm .* ((SWE1T ./ (1.0 + exp(( - sig .* Wswe1)))) - SWE1P));
	LumpedJ_reaction_51 = (((kd_swe1 + (kd_cdh1_swe1 .* CDH1A)) .* (SWE1T - SWE1P)) + ((kd_swe1p + (kd_cdh1_swe1p .* CDH1A)) .* SWE1P));
	VS2 = (PsS6S4Wp + (ks4s6p .* PsS6S4p) + (ks4s6 .* PsS6S4np) + (ks6m .* PsS6Mp) + (ks4 .* PsS4p));
	LumpedJ_reaction_50 = (ks_swe1 + (ks_sbf_swe1 .* VS2));
	S6M = ((((Swi6 .* Mbp1) ./ (Swi4 + Mbp1)) .* (Mbp1 >= ((Swi6 .* Mbp1) ./ (Swi4 + Mbp1)))) + (Mbp1 .* ~((Mbp1 >= ((Swi6 .* Mbp1) ./ (Swi4 + Mbp1))))));
	PmS6M = ((S6M .* ((0.85 .* Pm_0) >= S6M)) + ((0.85 .* Pm_0) .* ~(((0.85 .* Pm_0) >= S6M))));
	S6Mfree = (S6M - PmS6M);
	Wmcm1 = ((wa_mcm1_b2 .* CLB2) - wi_mcm1);
	MCM1 = (MCM1T ./ (1.0 + exp(( - sig .* Wmcm1))));
	CDC14 = ((0.0 .* (0.0 >= (CDC14T - (rho_14_net1 .* NET1A)))) + ((CDC14T - (rho_14_net1 .* NET1A)) .* ~((0.0 >= (CDC14T - (rho_14_net1 .* NET1A))))));
	Wbub2 = ((((wa_bub2_px .* CDC55) + (wa_bub2_14 .* CDC14) + wa_bub2) - (wi_bub2_lte_lo .* CDC5A .* I_SPOC)) - (wi_bub2_lo .* CDC5A));
	LumpedJ_reaction_46 = (gamm .* ((Bub2T ./ (1.0 + exp(( - sig .* Wbub2)))) - BUB2));
	Wpolo = ((wa_polo_b2 .* CLB2) - wi_polo);
	LumpedJ_reaction_45 = (gamm .* ((CDC5T ./ (1.0 + exp(( - sig .* Wpolo)))) - CDC5A));
	LumpedJ_reaction_44 = ((kd_polo + (kd_polo_h1 .* CDH1A)) .* CDC5T);
	LumpedJ_reaction_43 = (ks_polo + (ks_polo_m1 .* MCM1));
	Wtem1 = (((wa_tem1_polo .* CDC5A) + wa_tem1) - (wi_tem1_bub2 .* BUB2));
	LumpedJ_reaction_42 = (gammtem .* ((TEM1T ./ (1.0 + exp(( - sig .* Wtem1)))) - TEM1));
	Wcdc15 = (((wa_cdc15_14 .* CDC14) - wi_cdc15) - (wi_cdc15_b2 .* CLB2));
	LumpedJ_reaction_41 = (gamm .* ((CDC15T ./ (1.0 + exp(( - sigcdc15 .* Wcdc15)))) - CDC15));
	CDC15f = (((CDC15 - TEM1) .* ((CDC15 - TEM1) >= 0.0)) + (0.0 .* ~(((CDC15 - TEM1) >= 0.0))));
	MEN = (((1.0 .* (CDC5A >= 0.0)) + (0.0 .* ~((CDC5A >= 0.0)))) .* ((CDC15 .* (TEM1 >= CDC15)) + (TEM1 .* ~((TEM1 >= CDC15)))));
	CDC14n = (CDC14 - CDC14c);
	Wnet1 = (((wdp_net1_px .* CDC55) + wdp_net + (wdp_net1_14 .* CDC14n)) - (CDC5A .* (wp_net1 + (wp_net1_b2 .* CLB2) + (wp_net1_en .* MEN) + (wp_net1_15 .* CDC15f))));
	LumpedJ_reaction_40 = (gamm .* ((NET1T ./ (1.0 + exp(( - sig .* Wnet1)))) - NET1A));
	Psfree = ((0.0 .* (0.0 >= ((Ps_0 - PsS6S4) - PsS6S4W))) + (((Ps_0 - PsS6S4) - PsS6S4W) .* ~((0.0 >= ((Ps_0 - PsS6S4) - PsS6S4W)))));
	LumpedJ_reaction_39 = (k_cdc14_imp .* CDC14c);
	LumpedJ_reaction_38 = (k_cdc14_exp .* (CDC14 - CDC14c) .* MEN);
	Wcdc55 = ( - wi_ppx_p1 .* ESP1);
	LumpedJ_reaction_37 = (gamm .* ((CDC55T ./ (1.0 + exp(( - sig .* Wcdc55)))) - CDC55));
	CDC20A = ((0.0 .* (0.0 >= (CDC20T - MAD2A))) + ((CDC20T - MAD2A) .* ~((0.0 >= (CDC20T - MAD2A)))));
	CDC20A_APCP = ((APCP .* (CDC20A >= APCP)) + (CDC20A .* ~((CDC20A >= APCP))));
	CDC20A_APC = (((APCT - APCP) .* ((CDC20A - CDC20A_APCP) >= (APCT - APCP))) + ((CDC20A - CDC20A_APCP) .* ~(((CDC20A - CDC20A_APCP) >= (APCT - APCP)))));
	LumpedJ_reaction_36 = ((kd_pds + (kd_pds_20 .* CDC20A_APCP) + (kd_pds_20_i .* CDC20A_APC)) .* PDS1T);
	Pmfree = ((0.0 .* (0.0 >= ((Pm_0 - PmS6S4) - PmS6M))) + (((Pm_0 - PmS6S4) - PmS6M) .* ~((0.0 >= ((Pm_0 - PmS6S4) - PmS6M)))));
	VS5 = ((k_mbf_mp .* PmS6Mp) + (k_mbf_swi4 .* PmS6S4p) + ((Pmfree .* k_clb5 .* ks_clb5) ./ (ks_clb5_mbf + 1.0E-8)));
	LumpedJ_reaction_35 = (ks_pds1 + (ks_pds1_mbf .* VS5));
	CLB5 = ((CLB5T .* ((1.0 - ((CKIT .* h_sic1) ./ (CLB2T + CLB2nd + CLB5T))) .* (0.0 < (1.0 - ((CKIT .* h_sic1) ./ (CLB2T + CLB2nd + CLB5T)))))) .* ((CLB5T + CLB2T + CLB2nd) > 1.0E-8));
	Wcdh1 = ((wa_cdh1 + (wa_cdh1_14 .* CDC14)) - ((wi_cdh1_n2 .* CLN2) + (wi_cdh1_b5 .* CLB5) + (wi_cdh1_b2 .* CLB2)));
	SAC = (((UDNA .* (1.0 - SPNALIGN)) .* ((UDNA .* (1.0 - SPNALIGN)) >= HU)) + (HU .* ~(((UDNA .* (1.0 - SPNALIGN)) >= HU))));
	Wmad2 = ((wa_mad2 .* SAC) - wi_mad2);
	LumpedJ_reaction_34 = (gamm .* ((MAD2T ./ (1.0 + exp(( - sig .* Wmad2)))) - MAD2A));
	LumpedJ_reaction_33 = (kd_cdc20 .* CDC20T);
	LumpedJ_reaction_32 = (ks_cdc20 + (ks_cdc20_m1 .* MCM1));
	Wapc = ((wa_apc_b2 .* CLB2) - wi_apc);
	LumpedJ_reaction_31 = (gammcp .* ((APCT ./ (1.0 + exp(( - sig .* Wapc)))) - APCP));
	LumpedJ_reaction_30 = (gamm .* ((CDH1T ./ (1.0 + exp(( - sig .* Wcdh1)))) - CDH1A));
	Wssa1 = ((wssa1_clb2 .* CLB2) - wssa1);
	LumpedJ_reaction_29 = (kd_swi5 .* SWI5T);
	LumpedJ_reaction_28 = (ks_swi5 + (ks_swi5_m1 .* MCM1));
	LumpedJ_reaction_27 = (kd_spn .* SPN);
	LumpedJ_reaction_26 = (ks_spn .* ((1.0 .* ((CLB2 - Jspn) >= 0.0)) + (0.0 .* ~(((CLB2 - Jspn) >= 0.0)))));
	LumpedJ_reaction_25 = (kd_ori .* ORI);
	PsS4 = ((S4free .* (Psfree >= S4free)) + (Psfree .* ~((Psfree >= S4free))));
	LumpedJ_reaction_24 = (ks_ori_e .* ((e_ori_b5 .* CLB5) + (e_ori_b2 .* CLB2)));
	LumpedJ_reaction_23 = (kd_bud .* BUD);
	SSA1p = (SSA1T ./ (1.0 + exp(( - sig .* Wssa1))));
	Ydj1 = (kydj1 .* (1.0 - exp( - (MASS ./ kmass_ydj1))));
	SSA1free = ((((SSA1T - SSA1p) - Ydj1) .* (((SSA1T - SSA1p) - Ydj1) >= 0.0)) + (0.0 .* ~((((SSA1T - SSA1p) - Ydj1) >= 0.0))));
	CLN3n = (((CLN3 - (SSA1free + SSA1p)) .* ((CLN3 - (SSA1free + SSA1p)) >= 0.0)) + (0.0 .* ~(((CLN3 - (SSA1free + SSA1p)) >= 0.0))));
	LumpedJ_reaction_22 = (ks_bud_e .* ((e_bud_n3 .* CLN3n) + (e_bud_n2 .* CLN2) + (e_bud_b5 .* CLB5)));
	LumpedJ_reaction_21 = ((kd_clb2 + (kd_clb2_20 .* CDC20A_APCP) + (kd_clb2_20_i .* CDC20A_APC) + (kd_clb2_h1 .* CDH1A)) .* CLB2T);
	LumpedJ_reaction_20 = ((ks_clb2 + (ks_clb2_m1 .* MCM1)) .* (1.0 - exp( - (MASS ./ kmass_clb2))));
	SPNvsESP = (((1.0 .* ((ks_pds1 .* ks_pds1_mbf) >= 0.0)) + (0.0 .* ~(((ks_pds1 .* ks_pds1_mbf) >= 0.0)))) .* ((1.0 .* ((ESP1 - Theta_cleave) >= 0.0)) + (0.0 .* ~(((ESP1 - Theta_cleave) >= 0.0)))) .* ((1.0 .* ((1.0 - SPN) >= 0.0)) + (0.0 .* ~(((1.0 - SPN) >= 0.0)))) .* ((1.0 .* ((CLB2 - KEZ2) >= 0.0)) + (0.0 .* ~(((CLB2 - KEZ2) >= 0.0)))));
	Ws6s4w = ((((wp_s4s6pw_n3 .* CLN3n) + (wp_s4s6pw_k2 .* BCK2) + (wp_s4s6pw_n2 .* CLN2) + (wp_s4s6pw_b5 .* CLB5)) - wi_s4s6pw) - (CLB2 .* wp_s4s6pw_b2));
	LumpedJ_reaction_19 = ((kd_clb5 + (kd_clb5_20 .* CDC20A_APCP) + (kd_clb5_20_i .* CDC20A_APC)) .* CLB5T);
	LumpedJ_reaction_18 = ((ks_clb5 + (ks_clb5_mbf .* VS5)) .* (1.0 - exp( - (MASS ./ kmass_clb5))));
	LumpedJ_reaction_17 = (kd_kip .* CKIP);
	Wcki = ((((wp_cki_n2 .* CLN2) + (wp_cki_b5 .* CLB5) + (wp_cki_b2 .* CLB2)) - wdp_cki) - (wdp_cki_14 .* CDC14));
	LumpedJ_reaction_16 = (gammki .* ((CKIT ./ (1.0 + exp(( - sig .* Wcki)))) - CKIP));
	LumpedJ_reaction_15 = ((kd_ki .* (CKIT - CKIP)) + (kd_kip .* CKIP));
	Wswi5 = (( - wi_swi5_b2 .* CLB2) + wa_swi5 + (wa_swi5_14 .* CDC14));
	SWI5A = (SWI5T ./ (1.0 + exp(( - sig .* Wswi5))));
	LumpedJ_reaction_14 = (ks_cki + (ks_cki_swi5 .* SWI5A));
	LumpedJ_reaction_13 = (kd_n2 .* CLN2);
	LumpedJ_reaction_12 = (ks_cln2 + (ks_cln2_sbf .* VS2));
	Ws4 = (((wp_s4s4p_k2 .* BCK2) - wi_s4s4p) - (CLB2 .* wp_s4s4p_b2));
	LumpedJ_reaction_11 = (gamm .* ((PsS4 ./ (1.0 + exp(( - sig .* Ws4)))) - PsS4p));
	Ws6m = ((((wp_s6mb_n3 .* CLN3n) + (wp_s6mb_k2 .* BCK2) + (wp_s6mb_n2 .* CLN2) + (wp_s6mb_b5 .* CLB5)) - wi_s6mb) - (CLB2 .* wp_s6mb_b2));
	Psfree2 = ((0.0 .* (0.0 >= (((Ps_0 - PsS6S4) - PsS6S4W) - PsS4))) + ((((Ps_0 - PsS6S4) - PsS6S4W) - PsS4) .* ~((0.0 >= (((Ps_0 - PsS6S4) - PsS6S4W) - PsS4)))));
	PsS6M = ((S6Mfree .* (Psfree2 >= S6Mfree)) + (Psfree2 .* ~((Psfree2 >= S6Mfree))));
	LumpedJ_reaction_10 = (gamm .* ((PsS6M ./ (1.0 + exp(( - sig .* Ws6m)))) - PsS6Mp));
	Ws6s4np = ( - (((wpm_s6s4_n3 .* CLN3n) + (wpm_s6s4_k2 .* BCK2) + (wpm_s6s4_n2 .* CLN2) + (wpm_s6s4_b5 .* CLB5)) - wi_s6s4) - (CLB2 .* wp_s6s4_b2));
	Whi5free = (Whi5 - PsS6S4W);
	mu = (log(2.0) ./ MDT);
	UDNA1 = ((1.0 .* ((UDNA + HU) >= 0.0)) + (0.0 .* ~(((UDNA + HU) >= 0.0))));
	PmS4 = ((S4free .* ((0.15 .* Pmfree) >= S4free)) + ((0.15 .* Pmfree) .* ~(((0.15 .* Pmfree) >= S4free))));
	LumpedJ_reaction_9 = (gamm .* ((PsS6S4W ./ (1.0 + exp(( - sig .* Ws6s4w)))) - PsS6S4Wp));
	LumpedJ_reaction_8 = (gamm .* ((PsS6S4 ./ (1.0 + exp(( - sig .* Ws6s4np)))) - PsS6S4np));
	Ws6s4 = ((((wpm_s6s4_n3 .* CLN3n) + (wpm_s6s4_k2 .* BCK2) + (wpm_s6s4_n2 .* CLN2) + (wpm_s6s4_b5 .* CLB5)) - wi_s6s4) - (CLB2 .* wpm_s6s4_b2));
	LumpedJ_reaction_7 = (gamm .* ((PsS6S4 ./ (1.0 + exp(( - sig .* Ws6s4)))) - PsS6S4p));
	LumpedJ_reaction_6 = (gamm .* ((PmS6S4 ./ (1.0 + exp(( - sig .* Ws6s4)))) - PmS6S4p));
	LumpedJ_reaction_5 = (gamm .* ((PmS6M ./ (1.0 + exp(( - sig .* Ws6m)))) - PmS6Mp));
	LumpedJ_reaction_4 = (kd_k2 .* BCK2);
	LumpedJ_reaction_3 = (ks_k2 .* (1.0 - exp( - (MASS ./ kmass_bck2))));
	LumpedJ_reaction_2 = (kd_cln3 .* CLN3);
	LumpedJ_reaction_1 = (ks_n3 .* (1.0 - exp( - (MASS ./ kmass_cln3))));
	LumpedJ_reaction_0 = (mu .* MASS .* (1.0 - (MASS ./ Vmax)));
	% Rates
	dydt = [
		(LumpedJ_reaction_45 ./ Size_cell);    % rate for CDC5A
		(LumpedJ_reaction_37 ./ Size_cell);    % rate for CDC55
		(LumpedJ_reaction_49 ./ Size_cell);    % rate for UDNA
		(LumpedJ_reaction_7 ./ Size_cell);    % rate for PsS6S4p
		(LumpedJ_reaction_48 ./ Size_cell);    % rate for SPNALIGN
		(LumpedJ_reaction_41 ./ Size_cell);    % rate for CDC15
		((LumpedJ_reaction_35 ./ Size_cell) - (LumpedJ_reaction_36 ./ Size_cell));    % rate for PDS1T
		(LumpedJ_reaction_9 ./ Size_cell);    % rate for PsS6S4Wp
		((LumpedJ_reaction_1 ./ Size_cell) - (LumpedJ_reaction_2 ./ Size_cell));    % rate for CLN3
		((LumpedJ_reaction_12 ./ Size_cell) - (LumpedJ_reaction_13 ./ Size_cell));    % rate for CLN2
		(LumpedJ_reaction_46 ./ Size_cell);    % rate for BUB2
		(LumpedJ_reaction_10 ./ Size_cell);    % rate for PsS6Mp
		(LumpedJ_reaction_40 ./ Size_cell);    % rate for NET1A
		(LumpedJ_reaction_8 ./ Size_cell);    % rate for PsS6S4np
		(LumpedJ_reaction_31 ./ Size_cell);    % rate for APCP
		(LumpedJ_reaction_0 ./ Size_cell);    % rate for MASS
		((LumpedJ_reaction_28 ./ Size_cell) - (LumpedJ_reaction_29 ./ Size_cell));    % rate for SWI5T
		(LumpedJ_reaction_47 ./ Size_cell);    % rate for ORIFLAG
		((LumpedJ_reaction_3 ./ Size_cell) - (LumpedJ_reaction_4 ./ Size_cell));    % rate for BCK2
		((LumpedJ_reaction_18 ./ Size_cell) - (LumpedJ_reaction_19 ./ Size_cell));    % rate for CLB5T
		(LumpedJ_reaction_11 ./ Size_cell);    % rate for PsS4p
		((LumpedJ_reaction_50 ./ Size_cell) - (LumpedJ_reaction_51 ./ Size_cell));    % rate for SWE1T
		((LumpedJ_reaction_52 ./ Size_cell) - (LumpedJ_reaction_53 ./ Size_cell));    % rate for SWE1P
		((LumpedJ_reaction_24 ./ Size_cell) - (LumpedJ_reaction_25 ./ Size_cell));    % rate for ORI
		((LumpedJ_reaction_14 ./ Size_cell) - (LumpedJ_reaction_15 ./ Size_cell));    % rate for CKIT
		((LumpedJ_reaction_16 ./ Size_cell) - (LumpedJ_reaction_17 ./ Size_cell));    % rate for CKIP
		((LumpedJ_reaction_20 ./ Size_cell) - (LumpedJ_reaction_21 ./ Size_cell));    % rate for CLB2T
		((LumpedJ_reaction_26 ./ Size_cell) - (LumpedJ_reaction_27 ./ Size_cell));    % rate for SPN
		(LumpedJ_reaction_5 ./ Size_cell);    % rate for PmS6Mp
		(LumpedJ_reaction_34 ./ Size_cell);    % rate for MAD2A
		(LumpedJ_reaction_42 ./ Size_cell);    % rate for TEM1
		((LumpedJ_reaction_32 ./ Size_cell) - (LumpedJ_reaction_33 ./ Size_cell));    % rate for CDC20T
		((LumpedJ_reaction_22 ./ Size_cell) - (LumpedJ_reaction_23 ./ Size_cell));    % rate for BUD
		((LumpedJ_reaction_38 ./ Size_cell) - (LumpedJ_reaction_39 ./ Size_cell));    % rate for CDC14c
		(LumpedJ_reaction_30 ./ Size_cell);    % rate for CDH1A
		(LumpedJ_reaction_6 ./ Size_cell);    % rate for PmS6S4p
		((LumpedJ_reaction_43 ./ Size_cell) - (LumpedJ_reaction_44 ./ Size_cell));    % rate for CDC5T
	];
end
