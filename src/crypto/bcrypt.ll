; Copyright 2026 Piyush Katariya
;
; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
; You may obtain a copy of the License at
;
;     http://www.apache.org/licenses/LICENSE-2.0
;
; Unless required by applicable law or agreed to in writing, software
; distributed under the License is distributed on an "AS IS" BASIS,
; WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
; See the License for the specific language governing permissions and
; limitations under the License.

; bcrypt (Provos-Mazieres EksBlowfish) + the Blowfish cipher it is built on.
;
; DESIGN:
;   * Blowfish state = P-array (18 x i32) immediately followed by four S-boxes
;     (4 x 256 x i32). They are laid out CONTIGUOUSLY as one i32[1042] at ctx+0
;     (P = ctx_i32[0..17], S = ctx_i32[18..1041]); the pi-derived init table is
;     therefore a single 4168-byte memcpy, and key-schedule generation is a
;     single sweep writing words 0..1041. BLOWFISH_CTX_SIZE = 4168, align 16.
;   * The init constants are the fractional hexadecimal digits of pi (the
;     algorithm's definition), generated from first principles — NOT copied
;     from any implementation. Verified: the Blowfish core reproduces the
;     classic ECB test vectors, and the bcrypt wrapper reproduces published
;     $2 hashes bit-for-bit (this is the correctness gate).
;   * F(x) = ((S0[x>>24] + S1[x>>16 &255]) ^ S2[x>>8 &255]) + S3[x &255], all
;     mod 2^32. Feistel encipher is 16 rounds with the swap folded into the
;     round recurrence (t = L^P[i]; L' = R ^ F(t); R' = t), then the trailing
;     P[16]/P[17] whitening — no explicit swaps, no extra temporaries.
;   * stream2word reads 32-bit big-endian words cyclically from a byte buffer,
;     the primitive both the key mixing (P ^= key stream) and the salt mixing
;     (EksBlowfish data stream) share. bf_expand is the one routine behind the
;     plain key schedule (datalen=0) AND the salted ExpandKey (datalen=16).
;   * EksBlowfish setup: init pi state; ExpandKey(salt,key); then 2^cost times
;     { ExpandKey(0,key); ExpandKey(0,salt) }. The 2^cost sweeps are the whole
;     cost of bcrypt — each is ~521 Blowfish encipherments — so the cost knob
;     scales work exponentially exactly as intended.
;   * The final hash enciphers "OrpheanBeholderScryDoubt" (6 big-endian words /
;     3 blocks) 64 times, emits 23 bytes, and the output is the bcrypt-alphabet
;     base64 of salt (22 chars) + hash (31 chars) framed as "$2b$CC$...".
;   * All cipher arithmetic is plain wrapping i32 (NO nuw/nsw — a no-wrap flag
;     would be poison here, Hazard #1). i64 table indices carry nuw/nsw.
;
; HARDENING-TODO: fast, not constant-time. bcrypt has no secret-dependent
;   branches (the S-box lookups are data-dependent by design, as in every
;   Blowfish/bcrypt), but the key/salt/context buffers are not zeroized; add
;   zeroization + a side-channel review in the hardening phase.
;
; API (C ABI, nounwind):
;   ; Blowfish primitive (standard key schedule, ECB single block)
;   void universe_crypto_blowfish_init(ptr ctx, ptr key, i64 keylen)
;   void universe_crypto_blowfish_encrypt(ptr ctx, ptr in8, ptr out8)
;   ; bcrypt
;   void universe_crypto_bcrypt_raw(ptr pass, i64 passlen, ptr salt16,
;                                   i32 cost, ptr out23)
;   void universe_crypto_bcrypt(ptr pass, i64 passlen, ptr salt16,
;                               i32 cost, ptr out)   ; writes 60 chars + NUL

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare i32 @llvm.bswap.i32(i32)
declare i64 @llvm.umin.i64(i64, i64)

@bf.init  = private unnamed_addr constant [1042 x i32] [ i32 608135816, i32 2242054355, i32 320440878, i32 57701188, i32 2752067618, i32 698298832, i32 137296536, i32 3964562569, i32 1160258022, i32 953160567, i32 3193202383, i32 887688300, i32 3232508343, i32 3380367581, i32 1065670069, i32 3041331479, i32 2450970073, i32 2306472731, i32 3509652390, i32 2564797868, i32 805139163, i32 3491422135, i32 3101798381, i32 1780907670, i32 3128725573, i32 4046225305, i32 614570311, i32 3012652279, i32 134345442, i32 2240740374, i32 1667834072, i32 1901547113, i32 2757295779, i32 4103290238, i32 227898511, i32 1921955416, i32 1904987480, i32 2182433518, i32 2069144605, i32 3260701109, i32 2620446009, i32 720527379, i32 3318853667, i32 677414384, i32 3393288472, i32 3101374703, i32 2390351024, i32 1614419982, i32 1822297739, i32 2954791486, i32 3608508353, i32 3174124327, i32 2024746970, i32 1432378464, i32 3864339955, i32 2857741204, i32 1464375394, i32 1676153920, i32 1439316330, i32 715854006, i32 3033291828, i32 289532110, i32 2706671279, i32 2087905683, i32 3018724369, i32 1668267050, i32 732546397, i32 1947742710, i32 3462151702, i32 2609353502, i32 2950085171, i32 1814351708, i32 2050118529, i32 680887927, i32 999245976, i32 1800124847, i32 3300911131, i32 1713906067, i32 1641548236, i32 4213287313, i32 1216130144, i32 1575780402, i32 4018429277, i32 3917837745, i32 3693486850, i32 3949271944, i32 596196993, i32 3549867205, i32 258830323, i32 2213823033, i32 772490370, i32 2760122372, i32 1774776394, i32 2652871518, i32 566650946, i32 4142492826, i32 1728879713, i32 2882767088, i32 1783734482, i32 3629395816, i32 2517608232, i32 2874225571, i32 1861159788, i32 326777828, i32 3124490320, i32 2130389656, i32 2716951837, i32 967770486, i32 1724537150, i32 2185432712, i32 2364442137, i32 1164943284, i32 2105845187, i32 998989502, i32 3765401048, i32 2244026483, i32 1075463327, i32 1455516326, i32 1322494562, i32 910128902, i32 469688178, i32 1117454909, i32 936433444, i32 3490320968, i32 3675253459, i32 1240580251, i32 122909385, i32 2157517691, i32 634681816, i32 4142456567, i32 3825094682, i32 3061402683, i32 2540495037, i32 79693498, i32 3249098678, i32 1084186820, i32 1583128258, i32 426386531, i32 1761308591, i32 1047286709, i32 322548459, i32 995290223, i32 1845252383, i32 2603652396, i32 3431023940, i32 2942221577, i32 3202600964, i32 3727903485, i32 1712269319, i32 422464435, i32 3234572375, i32 1170764815, i32 3523960633, i32 3117677531, i32 1434042557, i32 442511882, i32 3600875718, i32 1076654713, i32 1738483198, i32 4213154764, i32 2393238008, i32 3677496056, i32 1014306527, i32 4251020053, i32 793779912, i32 2902807211, i32 842905082, i32 4246964064, i32 1395751752, i32 1040244610, i32 2656851899, i32 3396308128, i32 445077038, i32 3742853595, i32 3577915638, i32 679411651, i32 2892444358, i32 2354009459, i32 1767581616, i32 3150600392, i32 3791627101, i32 3102740896, i32 284835224, i32 4246832056, i32 1258075500, i32 768725851, i32 2589189241, i32 3069724005, i32 3532540348, i32 1274779536, i32 3789419226, i32 2764799539, i32 1660621633, i32 3471099624, i32 4011903706, i32 913787905, i32 3497959166, i32 737222580, i32 2514213453, i32 2928710040, i32 3937242737, i32 1804850592, i32 3499020752, i32 2949064160, i32 2386320175, i32 2390070455, i32 2415321851, i32 4061277028, i32 2290661394, i32 2416832540, i32 1336762016, i32 1754252060, i32 3520065937, i32 3014181293, i32 791618072, i32 3188594551, i32 3933548030, i32 2332172193, i32 3852520463, i32 3043980520, i32 413987798, i32 3465142937, i32 3030929376, i32 4245938359, i32 2093235073, i32 3534596313, i32 375366246, i32 2157278981, i32 2479649556, i32 555357303, i32 3870105701, i32 2008414854, i32 3344188149, i32 4221384143, i32 3956125452, i32 2067696032, i32 3594591187, i32 2921233993, i32 2428461, i32 544322398, i32 577241275, i32 1471733935, i32 610547355, i32 4027169054, i32 1432588573, i32 1507829418, i32 2025931657, i32 3646575487, i32 545086370, i32 48609733, i32 2200306550, i32 1653985193, i32 298326376, i32 1316178497, i32 3007786442, i32 2064951626, i32 458293330, i32 2589141269, i32 3591329599, i32 3164325604, i32 727753846, i32 2179363840, i32 146436021, i32 1461446943, i32 4069977195, i32 705550613, i32 3059967265, i32 3887724982, i32 4281599278, i32 3313849956, i32 1404054877, i32 2845806497, i32 146425753, i32 1854211946, i32 1266315497, i32 3048417604, i32 3681880366, i32 3289982499, i32 2909710000, i32 1235738493, i32 2632868024, i32 2414719590, i32 3970600049, i32 1771706367, i32 1449415276, i32 3266420449, i32 422970021, i32 1963543593, i32 2690192192, i32 3826793022, i32 1062508698, i32 1531092325, i32 1804592342, i32 2583117782, i32 2714934279, i32 4024971509, i32 1294809318, i32 4028980673, i32 1289560198, i32 2221992742, i32 1669523910, i32 35572830, i32 157838143, i32 1052438473, i32 1016535060, i32 1802137761, i32 1753167236, i32 1386275462, i32 3080475397, i32 2857371447, i32 1040679964, i32 2145300060, i32 2390574316, i32 1461121720, i32 2956646967, i32 4031777805, i32 4028374788, i32 33600511, i32 2920084762, i32 1018524850, i32 629373528, i32 3691585981, i32 3515945977, i32 2091462646, i32 2486323059, i32 586499841, i32 988145025, i32 935516892, i32 3367335476, i32 2599673255, i32 2839830854, i32 265290510, i32 3972581182, i32 2759138881, i32 3795373465, i32 1005194799, i32 847297441, i32 406762289, i32 1314163512, i32 1332590856, i32 1866599683, i32 4127851711, i32 750260880, i32 613907577, i32 1450815602, i32 3165620655, i32 3734664991, i32 3650291728, i32 3012275730, i32 3704569646, i32 1427272223, i32 778793252, i32 1343938022, i32 2676280711, i32 2052605720, i32 1946737175, i32 3164576444, i32 3914038668, i32 3967478842, i32 3682934266, i32 1661551462, i32 3294938066, i32 4011595847, i32 840292616, i32 3712170807, i32 616741398, i32 312560963, i32 711312465, i32 1351876610, i32 322626781, i32 1910503582, i32 271666773, i32 2175563734, i32 1594956187, i32 70604529, i32 3617834859, i32 1007753275, i32 1495573769, i32 4069517037, i32 2549218298, i32 2663038764, i32 504708206, i32 2263041392, i32 3941167025, i32 2249088522, i32 1514023603, i32 1998579484, i32 1312622330, i32 694541497, i32 2582060303, i32 2151582166, i32 1382467621, i32 776784248, i32 2618340202, i32 3323268794, i32 2497899128, i32 2784771155, i32 503983604, i32 4076293799, i32 907881277, i32 423175695, i32 432175456, i32 1378068232, i32 4145222326, i32 3954048622, i32 3938656102, i32 3820766613, i32 2793130115, i32 2977904593, i32 26017576, i32 3274890735, i32 3194772133, i32 1700274565, i32 1756076034, i32 4006520079, i32 3677328699, i32 720338349, i32 1533947780, i32 354530856, i32 688349552, i32 3973924725, i32 1637815568, i32 332179504, i32 3949051286, i32 53804574, i32 2852348879, i32 3044236432, i32 1282449977, i32 3583942155, i32 3416972820, i32 4006381244, i32 1617046695, i32 2628476075, i32 3002303598, i32 1686838959, i32 431878346, i32 2686675385, i32 1700445008, i32 1080580658, i32 1009431731, i32 832498133, i32 3223435511, i32 2605976345, i32 2271191193, i32 2516031870, i32 1648197032, i32 4164389018, i32 2548247927, i32 300782431, i32 375919233, i32 238389289, i32 3353747414, i32 2531188641, i32 2019080857, i32 1475708069, i32 455242339, i32 2609103871, i32 448939670, i32 3451063019, i32 1395535956, i32 2413381860, i32 1841049896, i32 1491858159, i32 885456874, i32 4264095073, i32 4001119347, i32 1565136089, i32 3898914787, i32 1108368660, i32 540939232, i32 1173283510, i32 2745871338, i32 3681308437, i32 4207628240, i32 3343053890, i32 4016749493, i32 1699691293, i32 1103962373, i32 3625875870, i32 2256883143, i32 3830138730, i32 1031889488, i32 3479347698, i32 1535977030, i32 4236805024, i32 3251091107, i32 2132092099, i32 1774941330, i32 1199868427, i32 1452454533, i32 157007616, i32 2904115357, i32 342012276, i32 595725824, i32 1480756522, i32 206960106, i32 497939518, i32 591360097, i32 863170706, i32 2375253569, i32 3596610801, i32 1814182875, i32 2094937945, i32 3421402208, i32 1082520231, i32 3463918190, i32 2785509508, i32 435703966, i32 3908032597, i32 1641649973, i32 2842273706, i32 3305899714, i32 1510255612, i32 2148256476, i32 2655287854, i32 3276092548, i32 4258621189, i32 236887753, i32 3681803219, i32 274041037, i32 1734335097, i32 3815195456, i32 3317970021, i32 1899903192, i32 1026095262, i32 4050517792, i32 356393447, i32 2410691914, i32 3873677099, i32 3682840055, i32 3913112168, i32 2491498743, i32 4132185628, i32 2489919796, i32 1091903735, i32 1979897079, i32 3170134830, i32 3567386728, i32 3557303409, i32 857797738, i32 1136121015, i32 1342202287, i32 507115054, i32 2535736646, i32 337727348, i32 3213592640, i32 1301675037, i32 2528481711, i32 1895095763, i32 1721773893, i32 3216771564, i32 62756741, i32 2142006736, i32 835421444, i32 2531993523, i32 1442658625, i32 3659876326, i32 2882144922, i32 676362277, i32 1392781812, i32 170690266, i32 3921047035, i32 1759253602, i32 3611846912, i32 1745797284, i32 664899054, i32 1329594018, i32 3901205900, i32 3045908486, i32 2062866102, i32 2865634940, i32 3543621612, i32 3464012697, i32 1080764994, i32 553557557, i32 3656615353, i32 3996768171, i32 991055499, i32 499776247, i32 1265440854, i32 648242737, i32 3940784050, i32 980351604, i32 3713745714, i32 1749149687, i32 3396870395, i32 4211799374, i32 3640570775, i32 1161844396, i32 3125318951, i32 1431517754, i32 545492359, i32 4268468663, i32 3499529547, i32 1437099964, i32 2702547544, i32 3433638243, i32 2581715763, i32 2787789398, i32 1060185593, i32 1593081372, i32 2418618748, i32 4260947970, i32 69676912, i32 2159744348, i32 86519011, i32 2512459080, i32 3838209314, i32 1220612927, i32 3339683548, i32 133810670, i32 1090789135, i32 1078426020, i32 1569222167, i32 845107691, i32 3583754449, i32 4072456591, i32 1091646820, i32 628848692, i32 1613405280, i32 3757631651, i32 526609435, i32 236106946, i32 48312990, i32 2942717905, i32 3402727701, i32 1797494240, i32 859738849, i32 992217954, i32 4005476642, i32 2243076622, i32 3870952857, i32 3732016268, i32 765654824, i32 3490871365, i32 2511836413, i32 1685915746, i32 3888969200, i32 1414112111, i32 2273134842, i32 3281911079, i32 4080962846, i32 172450625, i32 2569994100, i32 980381355, i32 4109958455, i32 2819808352, i32 2716589560, i32 2568741196, i32 3681446669, i32 3329971472, i32 1835478071, i32 660984891, i32 3704678404, i32 4045999559, i32 3422617507, i32 3040415634, i32 1762651403, i32 1719377915, i32 3470491036, i32 2693910283, i32 3642056355, i32 3138596744, i32 1364962596, i32 2073328063, i32 1983633131, i32 926494387, i32 3423689081, i32 2150032023, i32 4096667949, i32 1749200295, i32 3328846651, i32 309677260, i32 2016342300, i32 1779581495, i32 3079819751, i32 111262694, i32 1274766160, i32 443224088, i32 298511866, i32 1025883608, i32 3806446537, i32 1145181785, i32 168956806, i32 3641502830, i32 3584813610, i32 1689216846, i32 3666258015, i32 3200248200, i32 1692713982, i32 2646376535, i32 4042768518, i32 1618508792, i32 1610833997, i32 3523052358, i32 4130873264, i32 2001055236, i32 3610705100, i32 2202168115, i32 4028541809, i32 2961195399, i32 1006657119, i32 2006996926, i32 3186142756, i32 1430667929, i32 3210227297, i32 1314452623, i32 4074634658, i32 4101304120, i32 2273951170, i32 1399257539, i32 3367210612, i32 3027628629, i32 1190975929, i32 2062231137, i32 2333990788, i32 2221543033, i32 2438960610, i32 1181637006, i32 548689776, i32 2362791313, i32 3372408396, i32 3104550113, i32 3145860560, i32 296247880, i32 1970579870, i32 3078560182, i32 3769228297, i32 1714227617, i32 3291629107, i32 3898220290, i32 166772364, i32 1251581989, i32 493813264, i32 448347421, i32 195405023, i32 2709975567, i32 677966185, i32 3703036547, i32 1463355134, i32 2715995803, i32 1338867538, i32 1343315457, i32 2802222074, i32 2684532164, i32 233230375, i32 2599980071, i32 2000651841, i32 3277868038, i32 1638401717, i32 4028070440, i32 3237316320, i32 6314154, i32 819756386, i32 300326615, i32 590932579, i32 1405279636, i32 3267499572, i32 3150704214, i32 2428286686, i32 3959192993, i32 3461946742, i32 1862657033, i32 1266418056, i32 963775037, i32 2089974820, i32 2263052895, i32 1917689273, i32 448879540, i32 3550394620, i32 3981727096, i32 150775221, i32 3627908307, i32 1303187396, i32 508620638, i32 2975983352, i32 2726630617, i32 1817252668, i32 1876281319, i32 1457606340, i32 908771278, i32 3720792119, i32 3617206836, i32 2455994898, i32 1729034894, i32 1080033504, i32 976866871, i32 3556439503, i32 2881648439, i32 1522871579, i32 1555064734, i32 1336096578, i32 3548522304, i32 2579274686, i32 3574697629, i32 3205460757, i32 3593280638, i32 3338716283, i32 3079412587, i32 564236357, i32 2993598910, i32 1781952180, i32 1464380207, i32 3163844217, i32 3332601554, i32 1699332808, i32 1393555694, i32 1183702653, i32 3581086237, i32 1288719814, i32 691649499, i32 2847557200, i32 2895455976, i32 3193889540, i32 2717570544, i32 1781354906, i32 1676643554, i32 2592534050, i32 3230253752, i32 1126444790, i32 2770207658, i32 2633158820, i32 2210423226, i32 2615765581, i32 2414155088, i32 3127139286, i32 673620729, i32 2805611233, i32 1269405062, i32 4015350505, i32 3341807571, i32 4149409754, i32 1057255273, i32 2012875353, i32 2162469141, i32 2276492801, i32 2601117357, i32 993977747, i32 3918593370, i32 2654263191, i32 753973209, i32 36408145, i32 2530585658, i32 25011837, i32 3520020182, i32 2088578344, i32 530523599, i32 2918365339, i32 1524020338, i32 1518925132, i32 3760827505, i32 3759777254, i32 1202760957, i32 3985898139, i32 3906192525, i32 674977740, i32 4174734889, i32 2031300136, i32 2019492241, i32 3983892565, i32 4153806404, i32 3822280332, i32 352677332, i32 2297720250, i32 60907813, i32 90501309, i32 3286998549, i32 1016092578, i32 2535922412, i32 2839152426, i32 457141659, i32 509813237, i32 4120667899, i32 652014361, i32 1966332200, i32 2975202805, i32 55981186, i32 2327461051, i32 676427537, i32 3255491064, i32 2882294119, i32 3433927263, i32 1307055953, i32 942726286, i32 933058658, i32 2468411793, i32 3933900994, i32 4215176142, i32 1361170020, i32 2001714738, i32 2830558078, i32 3274259782, i32 1222529897, i32 1679025792, i32 2729314320, i32 3714953764, i32 1770335741, i32 151462246, i32 3013232138, i32 1682292957, i32 1483529935, i32 471910574, i32 1539241949, i32 458788160, i32 3436315007, i32 1807016891, i32 3718408830, i32 978976581, i32 1043663428, i32 3165965781, i32 1927990952, i32 4200891579, i32 2372276910, i32 3208408903, i32 3533431907, i32 1412390302, i32 2931980059, i32 4132332400, i32 1947078029, i32 3881505623, i32 4168226417, i32 2941484381, i32 1077988104, i32 1320477388, i32 886195818, i32 18198404, i32 3786409000, i32 2509781533, i32 112762804, i32 3463356488, i32 1866414978, i32 891333506, i32 18488651, i32 661792760, i32 1628790961, i32 3885187036, i32 3141171499, i32 876946877, i32 2693282273, i32 1372485963, i32 791857591, i32 2686433993, i32 3759982718, i32 3167212022, i32 3472953795, i32 2716379847, i32 445679433, i32 3561995674, i32 3504004811, i32 3574258232, i32 54117162, i32 3331405415, i32 2381918588, i32 3769707343, i32 4154350007, i32 1140177722, i32 4074052095, i32 668550556, i32 3214352940, i32 367459370, i32 261225585, i32 2610173221, i32 4209349473, i32 3468074219, i32 3265815641, i32 314222801, i32 3066103646, i32 3808782860, i32 282218597, i32 3406013506, i32 3773591054, i32 379116347, i32 1285071038, i32 846784868, i32 2669647154, i32 3771962079, i32 3550491691, i32 2305946142, i32 453669953, i32 1268987020, i32 3317592352, i32 3279303384, i32 3744833421, i32 2610507566, i32 3859509063, i32 266596637, i32 3847019092, i32 517658769, i32 3462560207, i32 3443424879, i32 370717030, i32 4247526661, i32 2224018117, i32 4143653529, i32 4112773975, i32 2788324899, i32 2477274417, i32 1456262402, i32 2901442914, i32 1517677493, i32 1846949527, i32 2295493580, i32 3734397586, i32 2176403920, i32 1280348187, i32 1908823572, i32 3871786941, i32 846861322, i32 1172426758, i32 3287448474, i32 3383383037, i32 1655181056, i32 3139813346, i32 901632758, i32 1897031941, i32 2986607138, i32 3066810236, i32 3447102507, i32 1393639104, i32 373351379, i32 950779232, i32 625454576, i32 3124240540, i32 4148612726, i32 2007998917, i32 544563296, i32 2244738638, i32 2330496472, i32 2058025392, i32 1291430526, i32 424198748, i32 50039436, i32 29584100, i32 3605783033, i32 2429876329, i32 2791104160, i32 1057563949, i32 3255363231, i32 3075367218, i32 3463963227, i32 1469046755, i32 985887462 ], align 16
@bf.magic = private unnamed_addr constant [24 x i8] c"OrpheanBeholderScryDoubt"
@bf.b64   = private unnamed_addr constant [64 x i8] c"./ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

; ---------------------------------------------------------------- Feistel F
define internal i32 @bf_f(ptr %ctx, i32 %x) #0 {
entry:
  %S = getelementptr inbounds nuw i32, ptr %ctx, i64 18
  %a = lshr i32 %x, 24
  %ai = zext i32 %a to i64
  %pa = getelementptr inbounds nuw i32, ptr %S, i64 %ai
  %sa = load i32, ptr %pa, align 4
  %b0 = lshr i32 %x, 16
  %b = and i32 %b0, 255
  %bi0 = zext i32 %b to i64
  %bi = add nuw nsw i64 %bi0, 256
  %pb = getelementptr inbounds nuw i32, ptr %S, i64 %bi
  %sb = load i32, ptr %pb, align 4
  %c0 = lshr i32 %x, 8
  %c = and i32 %c0, 255
  %ci0 = zext i32 %c to i64
  %ci = add nuw nsw i64 %ci0, 512
  %pc = getelementptr inbounds nuw i32, ptr %S, i64 %ci
  %sc = load i32, ptr %pc, align 4
  %d = and i32 %x, 255
  %di0 = zext i32 %d to i64
  %di = add nuw nsw i64 %di0, 768
  %pd = getelementptr inbounds nuw i32, ptr %S, i64 %di
  %sd = load i32, ptr %pd, align 4
  %s1 = add i32 %sa, %sb
  %s2 = xor i32 %s1, %sc
  %s3 = add i32 %s2, %sd
  ret i32 %s3
}

; ------------------------------------------------------------- encipher block
; Returns (L<<32)|R after 16-round Blowfish encryption of (L0,R0).
define internal i64 @bf_encipher(ptr %ctx, i32 %L0, i32 %R0) #1 {
entry:
  br label %rh

rh:
  %i = phi i64 [ 0, %entry ], [ %i.n, %rb ]
  %L = phi i32 [ %L0, %entry ], [ %R1, %rb ]
  %R = phi i32 [ %R0, %entry ], [ %L1, %rb ]
  %ic = icmp ult i64 %i, 16
  br i1 %ic, label %rb, label %rx

rb:
  %pp = getelementptr inbounds nuw i32, ptr %ctx, i64 %i
  %pv = load i32, ptr %pp, align 4
  %L1 = xor i32 %L, %pv
  %f = call i32 @bf_f(ptr %ctx, i32 %L1)
  %R1 = xor i32 %f, %R
  %i.n = add nuw nsw i64 %i, 1
  br label %rh

rx:
  %p16p = getelementptr inbounds nuw i32, ptr %ctx, i64 16
  %p16 = load i32, ptr %p16p, align 4
  %p17p = getelementptr inbounds nuw i32, ptr %ctx, i64 17
  %p17 = load i32, ptr %p17p, align 4
  %fL = xor i32 %R, %p17
  %fR = xor i32 %L, %p16
  %hi = zext i32 %fL to i64
  %his = shl nuw i64 %hi, 32
  %lo = zext i32 %fR to i64
  %res = or i64 %his, %lo
  ret i64 %res
}

; ------------------------------------------------------------- stream2word
; Read a 32-bit big-endian word cyclically from buf[len] starting at idx.
; Returns { word, next_idx }. word=0 / idx unchanged when len==0.
define internal { i32, i64 } @bf_stream_word(ptr %buf, i64 %len, i64 %idx0) #0 {
entry:
  %z = icmp eq i64 %len, 0
  br i1 %z, label %zero, label %loop

zero:
  %r0a = insertvalue { i32, i64 } undef, i32 0, 0
  %r0b = insertvalue { i32, i64 } %r0a, i64 %idx0, 1
  ret { i32, i64 } %r0b

loop:
  %k = phi i64 [ 0, %entry ], [ %k.n, %loop ]
  %idx = phi i64 [ %idx0, %entry ], [ %idx.n, %loop ]
  %w = phi i32 [ 0, %entry ], [ %w.n, %loop ]
  %bp = getelementptr inbounds nuw i8, ptr %buf, i64 %idx
  %bv = load i8, ptr %bp, align 1
  %bz = zext i8 %bv to i32
  %ws = shl i32 %w, 8
  %w.n = or i32 %ws, %bz
  %idx1 = add nuw nsw i64 %idx, 1
  %wrap = icmp uge i64 %idx1, %len
  %idx.n = select i1 %wrap, i64 0, i64 %idx1
  %k.n = add nuw nsw i64 %k, 1
  %kc = icmp ult i64 %k.n, 4
  br i1 %kc, label %loop, label %done

done:
  %r1a = insertvalue { i32, i64 } undef, i32 %w.n, 0
  %r1b = insertvalue { i32, i64 } %r1a, i64 %idx.n, 1
  ret { i32, i64 } %r1b
}

; ------------------------------------------------------------- init pi state
define internal void @bf_init_state(ptr %ctx) #1 {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr %ctx, ptr @bf.init, i64 4168, i1 false)
  ret void
}

; ------------------------------------------------------------- expand key
; P ^= key stream; then regenerate P and S by enciphering a running block,
; XORing successive data words in first (datalen=0 => no data XOR).
define internal void @bf_expand(ptr %ctx, ptr %data, i64 %datalen, ptr %key, i64 %keylen) #1 {
entry:
  br label %ph

ph:
  %i = phi i64 [ 0, %entry ], [ %i.n, %pb ]
  %kidx = phi i64 [ 0, %entry ], [ %kidx.n, %pb ]
  %ic = icmp ult i64 %i, 18
  br i1 %ic, label %pb, label %gen.pre

pb:
  %sw = call { i32, i64 } @bf_stream_word(ptr %key, i64 %keylen, i64 %kidx)
  %kw = extractvalue { i32, i64 } %sw, 0
  %kidx.n = extractvalue { i32, i64 } %sw, 1
  %pp = getelementptr inbounds nuw i32, ptr %ctx, i64 %i
  %pv = load i32, ptr %pp, align 4
  %px = xor i32 %pv, %kw
  store i32 %px, ptr %pp, align 4
  %i.n = add nuw nsw i64 %i, 1
  br label %ph

gen.pre:
  %hasdata = icmp ne i64 %datalen, 0
  br label %gh

gh:
  %w = phi i64 [ 0, %gen.pre ], [ %w.n, %gb.enc ]
  %L = phi i32 [ 0, %gen.pre ], [ %Lo, %gb.enc ]
  %R = phi i32 [ 0, %gen.pre ], [ %Ro, %gb.enc ]
  %didx = phi i64 [ 0, %gen.pre ], [ %didxp, %gb.enc ]
  %wc = icmp ult i64 %w, 1042
  br i1 %wc, label %gb, label %ret

gb:
  br i1 %hasdata, label %gb.data, label %gb.zero

gb.data:
  %sd1 = call { i32, i64 } @bf_stream_word(ptr %data, i64 %datalen, i64 %didx)
  %dwl = extractvalue { i32, i64 } %sd1, 0
  %didx1 = extractvalue { i32, i64 } %sd1, 1
  %sd2 = call { i32, i64 } @bf_stream_word(ptr %data, i64 %datalen, i64 %didx1)
  %dwr = extractvalue { i32, i64 } %sd2, 0
  %didx2d = extractvalue { i32, i64 } %sd2, 1
  br label %gb.enc

gb.zero:
  br label %gb.enc

gb.enc:
  %dl = phi i32 [ %dwl, %gb.data ], [ 0, %gb.zero ]
  %dr = phi i32 [ %dwr, %gb.data ], [ 0, %gb.zero ]
  %didxp = phi i64 [ %didx2d, %gb.data ], [ %didx, %gb.zero ]
  %Lx = xor i32 %L, %dl
  %Rx = xor i32 %R, %dr
  %enc = call i64 @bf_encipher(ptr %ctx, i32 %Lx, i32 %Rx)
  %ehi = lshr i64 %enc, 32
  %Lo = trunc i64 %ehi to i32
  %Ro = trunc i64 %enc to i32
  %pw = getelementptr inbounds nuw i32, ptr %ctx, i64 %w
  store i32 %Lo, ptr %pw, align 4
  %w1 = add nuw nsw i64 %w, 1
  %pw1 = getelementptr inbounds nuw i32, ptr %ctx, i64 %w1
  store i32 %Ro, ptr %pw1, align 4
  %w.n = add nuw nsw i64 %w, 2
  br label %gh

ret:
  ret void
}

; ------------------------------------------- Blowfish public: init + encrypt
define void @universe_crypto_blowfish_init(ptr %ctx, ptr %key, i64 %keylen) local_unnamed_addr #1 {
entry:
  call void @bf_init_state(ptr %ctx)
  call void @bf_expand(ptr %ctx, ptr null, i64 0, ptr %key, i64 %keylen)
  ret void
}

define void @universe_crypto_blowfish_encrypt(ptr %ctx, ptr %in, ptr %out) local_unnamed_addr #1 {
entry:
  %l0 = load i32, ptr %in, align 1
  %L = call i32 @llvm.bswap.i32(i32 %l0)
  %inr = getelementptr inbounds nuw i8, ptr %in, i64 4
  %r0 = load i32, ptr %inr, align 1
  %R = call i32 @llvm.bswap.i32(i32 %r0)
  %enc = call i64 @bf_encipher(ptr %ctx, i32 %L, i32 %R)
  %ehi = lshr i64 %enc, 32
  %Lo = trunc i64 %ehi to i32
  %Ro = trunc i64 %enc to i32
  %Lob = call i32 @llvm.bswap.i32(i32 %Lo)
  store i32 %Lob, ptr %out, align 1
  %outr = getelementptr inbounds nuw i8, ptr %out, i64 4
  %Rob = call i32 @llvm.bswap.i32(i32 %Ro)
  store i32 %Rob, ptr %outr, align 1
  ret void
}

; ------------------------------------------------------ EksBlowfish setup
define internal void @bcrypt_eks_setup(ptr %ctx, i32 %cost, ptr %salt, ptr %key, i64 %keylen) #1 {
entry:
  call void @bf_init_state(ptr %ctx)
  call void @bf_expand(ptr %ctx, ptr %salt, i64 16, ptr %key, i64 %keylen)
  %cz = zext i32 %cost to i64
  %rounds = shl i64 1, %cz
  br label %lh

lh:
  %k = phi i64 [ 0, %entry ], [ %k.n, %lb ]
  %kc = icmp ult i64 %k, %rounds
  br i1 %kc, label %lb, label %ret

lb:
  call void @bf_expand(ptr %ctx, ptr null, i64 0, ptr %key, i64 %keylen)
  call void @bf_expand(ptr %ctx, ptr null, i64 0, ptr %salt, i64 16)
  %k.n = add nuw nsw i64 %k, 1
  br label %lh

ret:
  ret void
}

; ------------------------------------------------------------- bcrypt raw
define void @universe_crypto_bcrypt_raw(ptr %pass, i64 %passlen, ptr %salt, i32 %cost, ptr %out) local_unnamed_addr #1 {
entry:
  ; key = password (capped 72) + NUL terminator
  %kb = alloca [73 x i8], align 1
  %cap = call i64 @llvm.umin.i64(i64 %passlen, i64 72)
  call void @llvm.memcpy.p0.p0.i64(ptr %kb, ptr %pass, i64 %cap, i1 false)
  %np = getelementptr inbounds nuw i8, ptr %kb, i64 %cap
  store i8 0, ptr %np, align 1
  %keylen = add nuw nsw i64 %cap, 1
  %ctx = alloca [4168 x i8], align 16
  call void @bcrypt_eks_setup(ptr %ctx, i32 %cost, ptr %salt, ptr %kb, i64 %keylen)
  ; load magic into cdata[0..5]
  %cdata = alloca [6 x i32], align 16
  br label %ld.h

ld.h:
  %i = phi i64 [ 0, %entry ], [ %i.n, %ld.b ]
  %idx = phi i64 [ 0, %entry ], [ %idx.n, %ld.b ]
  %ic = icmp ult i64 %i, 6
  br i1 %ic, label %ld.b, label %r64.h

ld.b:
  %sw = call { i32, i64 } @bf_stream_word(ptr @bf.magic, i64 24, i64 %idx)
  %cw = extractvalue { i32, i64 } %sw, 0
  %idx.n = extractvalue { i32, i64 } %sw, 1
  %cp = getelementptr inbounds nuw [6 x i32], ptr %cdata, i64 0, i64 %i
  store i32 %cw, ptr %cp, align 4
  %i.n = add nuw nsw i64 %i, 1
  br label %ld.h

r64.h:
  %rr = phi i64 [ 0, %ld.h ], [ %rr.n, %r64.next ]
  %rc = icmp ult i64 %rr, 64
  br i1 %rc, label %blk.h, label %emit

blk.h:
  %j = phi i64 [ 0, %r64.h ], [ %j.n, %blk.b ]
  %jc = icmp ult i64 %j, 3
  br i1 %jc, label %blk.b, label %r64.next

blk.b:
  %j2 = shl nuw nsw i64 %j, 1
  %pL = getelementptr inbounds nuw [6 x i32], ptr %cdata, i64 0, i64 %j2
  %Lv = load i32, ptr %pL, align 4
  %j2p1 = or disjoint i64 %j2, 1
  %pR = getelementptr inbounds nuw [6 x i32], ptr %cdata, i64 0, i64 %j2p1
  %Rv = load i32, ptr %pR, align 4
  %enc = call i64 @bf_encipher(ptr %ctx, i32 %Lv, i32 %Rv)
  %ehi = lshr i64 %enc, 32
  %Lo = trunc i64 %ehi to i32
  %Ro = trunc i64 %enc to i32
  store i32 %Lo, ptr %pL, align 4
  store i32 %Ro, ptr %pR, align 4
  %j.n = add nuw nsw i64 %j, 1
  br label %blk.h

r64.next:
  %rr.n = add nuw nsw i64 %rr, 1
  br label %r64.h

emit:
  ; write first 5 words (20 bytes) then top 3 bytes of word 5 => 23 bytes
  br label %o.h

o.h:
  %oi = phi i64 [ 0, %emit ], [ %oi.n, %o.b ]
  %oc = icmp ult i64 %oi, 5
  br i1 %oc, label %o.b, label %tail

o.b:
  %wp = getelementptr inbounds nuw [6 x i32], ptr %cdata, i64 0, i64 %oi
  %wv = load i32, ptr %wp, align 4
  %wb = call i32 @llvm.bswap.i32(i32 %wv)
  %ob = shl nuw nsw i64 %oi, 2
  %op = getelementptr inbounds nuw i8, ptr %out, i64 %ob
  store i32 %wb, ptr %op, align 1
  %oi.n = add nuw nsw i64 %oi, 1
  br label %o.h

tail:
  %w5p = getelementptr inbounds nuw [6 x i32], ptr %cdata, i64 0, i64 5
  %w5 = load i32, ptr %w5p, align 4
  %b0 = lshr i32 %w5, 24
  %b0t = trunc i32 %b0 to i8
  %o20 = getelementptr inbounds nuw i8, ptr %out, i64 20
  store i8 %b0t, ptr %o20, align 1
  %b1 = lshr i32 %w5, 16
  %b1t = trunc i32 %b1 to i8
  %o21 = getelementptr inbounds nuw i8, ptr %out, i64 21
  store i8 %b1t, ptr %o21, align 1
  %b2 = lshr i32 %w5, 8
  %b2t = trunc i32 %b2 to i8
  %o22 = getelementptr inbounds nuw i8, ptr %out, i64 22
  store i8 %b2t, ptr %o22, align 1
  ret void
}

; ------------------------------------------------- bcrypt base64 (custom)
; Encode len bytes of data into dst using the bcrypt alphabet, no padding.
; Returns the number of characters written.
define internal i64 @bf_b64_encode(ptr %dst, ptr %data, i64 %len) #1 {
entry:
  br label %lh

lh:
  %i = phi i64 [ 0, %entry ], [ %i1, %e1 ], [ %i2, %e2 ], [ %i3, %cont2 ]
  %p = phi i64 [ 0, %entry ], [ %pe1, %e1 ], [ %pe2, %e2 ], [ %p4, %cont2 ]
  %ic = icmp ult i64 %i, %len
  br i1 %ic, label %body, label %done

body:
  %d0p = getelementptr inbounds nuw i8, ptr %data, i64 %i
  %d0 = load i8, ptr %d0p, align 1
  %c1 = zext i8 %d0 to i32
  %i1 = add nuw nsw i64 %i, 1
  ; emit b64[c1>>2]
  %e0i = lshr i32 %c1, 2
  call void @bf_b64_put(ptr %dst, i64 %p, i32 %e0i)
  %p1 = add nuw nsw i64 %p, 1
  %t0 = and i32 %c1, 3
  %t = shl nuw nsw i32 %t0, 4
  %end1 = icmp uge i64 %i1, %len
  br i1 %end1, label %e1, label %cont1

e1:
  call void @bf_b64_put(ptr %dst, i64 %p1, i32 %t)
  %pe1 = add nuw nsw i64 %p1, 1
  br label %lh

cont1:
  %d1p = getelementptr inbounds nuw i8, ptr %data, i64 %i1
  %d1 = load i8, ptr %d1p, align 1
  %c2 = zext i8 %d1 to i32
  %i2 = add nuw nsw i64 %i1, 1
  %c2hi = lshr i32 %c2, 4
  %c2hi4 = and i32 %c2hi, 15
  %t2 = or i32 %t, %c2hi4
  call void @bf_b64_put(ptr %dst, i64 %p1, i32 %t2)
  %p2 = add nuw nsw i64 %p1, 1
  %u0 = and i32 %c2, 15
  %u = shl nuw nsw i32 %u0, 2
  %end2 = icmp uge i64 %i2, %len
  br i1 %end2, label %e2, label %cont2

e2:
  call void @bf_b64_put(ptr %dst, i64 %p2, i32 %u)
  %pe2 = add nuw nsw i64 %p2, 1
  br label %lh

cont2:
  %d2p = getelementptr inbounds nuw i8, ptr %data, i64 %i2
  %d2 = load i8, ptr %d2p, align 1
  %c3 = zext i8 %d2 to i32
  %i3 = add nuw nsw i64 %i2, 1
  %c3hi = lshr i32 %c3, 6
  %c3hi2 = and i32 %c3hi, 3
  %v = or i32 %u, %c3hi2
  call void @bf_b64_put(ptr %dst, i64 %p2, i32 %v)
  %p3 = add nuw nsw i64 %p2, 1
  %c3lo = and i32 %c3, 63
  call void @bf_b64_put(ptr %dst, i64 %p3, i32 %c3lo)
  %p4 = add nuw nsw i64 %p3, 1
  br label %lh

done:
  ret i64 %p
}

; store bcrypt-alphabet char for the low 6 bits of %val at dst[pos]
define internal void @bf_b64_put(ptr %dst, i64 %pos, i32 %val) #0 {
entry:
  %v6 = and i32 %val, 63
  %vi = zext i32 %v6 to i64
  %cp = getelementptr inbounds nuw [64 x i8], ptr @bf.b64, i64 0, i64 %vi
  %ch = load i8, ptr %cp, align 1
  %dp = getelementptr inbounds nuw i8, ptr %dst, i64 %pos
  store i8 %ch, ptr %dp, align 1
  ret void
}

; ------------------------------------------------------------- bcrypt string
define void @universe_crypto_bcrypt(ptr %pass, i64 %passlen, ptr %salt, i32 %cost, ptr %out) local_unnamed_addr #1 {
entry:
  %raw = alloca [23 x i8], align 8
  call void @universe_crypto_bcrypt_raw(ptr %pass, i64 %passlen, ptr %salt, i32 %cost, ptr %raw)
  ; framing "$2b$CC$"
  store i8 36, ptr %out, align 1                 ; '$'
  %o1 = getelementptr inbounds nuw i8, ptr %out, i64 1
  store i8 50, ptr %o1, align 1                  ; '2'
  %o2 = getelementptr inbounds nuw i8, ptr %out, i64 2
  store i8 98, ptr %o2, align 1                  ; 'b'
  %o3 = getelementptr inbounds nuw i8, ptr %out, i64 3
  store i8 36, ptr %o3, align 1                  ; '$'
  %tens = udiv i32 %cost, 10
  %tensc0 = add nuw nsw i32 %tens, 48
  %tensc = trunc i32 %tensc0 to i8
  %o4 = getelementptr inbounds nuw i8, ptr %out, i64 4
  store i8 %tensc, ptr %o4, align 1
  %ones = urem i32 %cost, 10
  %onesc0 = add nuw nsw i32 %ones, 48
  %onesc = trunc i32 %onesc0 to i8
  %o5 = getelementptr inbounds nuw i8, ptr %out, i64 5
  store i8 %onesc, ptr %o5, align 1
  %o6 = getelementptr inbounds nuw i8, ptr %out, i64 6
  store i8 36, ptr %o6, align 1                  ; '$'
  ; salt base64 (16 bytes -> 22 chars) at out+7
  %sdst = getelementptr inbounds nuw i8, ptr %out, i64 7
  %sn = call i64 @bf_b64_encode(ptr %sdst, ptr %salt, i64 16)
  ; hash base64 (23 bytes -> 31 chars) at out+29
  %hdst = getelementptr inbounds nuw i8, ptr %out, i64 29
  %hn = call i64 @bf_b64_encode(ptr %hdst, ptr %raw, i64 23)
  ; NUL terminate at out+60
  %oend = getelementptr inbounds nuw i8, ptr %out, i64 60
  store i8 0, ptr %oend, align 1
  ret void
}

attributes #0 = { alwaysinline nounwind willreturn norecurse nosync nofree }
attributes #1 = { nounwind willreturn norecurse nosync nofree }
