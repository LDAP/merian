#pragma once

implementing colorspaces;

#include "merian-shaders/utils/math-constants.slangh"

/*
Adapted from VKDT, licensed under:

copyright 2019 johannes hanika

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions
and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the
distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

namespace merian {

// was inlined by the compiler
// static const int munsell_hdim = 40;
// static const int munsell_cdim = 21;

static const uint munsell_lut[840] = {
    893596928u, 888485006u,  882914332u,  877278059u,  871117480u,  860434938u,  850145621u,
    840839360u, 831598641u,  824389517u,  816721591u,  810495460u,  804007194u,  792800344u,
    782117694u, 771762649u,  761800831u,  752166499u,  742794164u,  729161098u,  711433233u,
    893596928u, 889402502u,  884356109u,  879113022u,  873869940u,  865087934u,  854733069u,
    845623410u, 836251577u,  828321478u,  822095366u,  816917821u,  811936894u,  807218062u,
    799943213u, 791161049u,  782575390u,  774317213u,  766255197u,  758419961u,  750756095u,
    893596928u, 890320002u,  886387716u,  881668897u,  876753480u,  871838086u,  862269639u,
    853880872u, 845098763u,  837103107u,  830352689u,  825306190u,  820456173u,  815868240u,
    811476871u, 807216265u,  800857751u,  792894620u,  785163198u,  777627269u,  770287756u,
    893596928u, 891434112u,  888419331u,  885076762u,  881472060u,  877801832u,  874000542u,
    869412869u, 862531262u,  856698297u,  850013328u,  845163215u,  840575136u,  836248860u,
    832117928u, 828154171u,  824354368u,  820685024u,  817146514u,  813673499u,  810331368u,
    893596928u, 892482689u,  890778635u,  888615724u,  886256205u,  883700086u,  881143985u,
    879046681u, 876621529u,  874196389u,  871705777u,  868494237u,  865806980u,  864102751u,
    862464066u, 860890714u,  859317310u,  857808991u,  856363422u,  854957865u,  853582702u,
    893596928u, 893858952u,  893793306u,  893334362u,  892416637u,  891236774u,  889925850u,
    888811590u, 887566138u,  886255099u,  885009652u,  884026403u,  882977550u,  882453018u,
    882125096u, 881862716u,  881534634u,  881272039u,  881009244u,  880680438u,  880447825u,
    893596928u, 895169683u,  896414762u,  897266577u,  897790645u,  898052577u,  898248984u,
    898314379u, 898117559u,  897789541u,  897592688u,  897133724u,  896936888u,  896477824u,
    896280939u, 896084053u,  896149133u,  896214143u,  896278779u,  896342085u,  896441732u,
    893596928u, 896808099u,  899691589u,  902181848u,  904737546u,  906834507u,  908800379u,
    910110949u, 911487025u,  912600874u,  913583640u,  914238753u,  914893902u,  915221324u,
    915876305u, 916269142u,  916989090u,  918166340u,  919314767u,  920364563u,  921413807u,
    893596928u, 897856687u,  901985370u,  905786376u,  909914964u,  913453736u,  917516761u,
    920334655u, 923349139u,  926167005u,  928329429u,  930360770u,  932195527u,  933768178u,
    935209567u, 937306166u,  939598652u,  940939704u,  942187732u,  943368288u,  944482881u,
    893596928u, 899036349u,  904410228u,  909456424u,  914830242u,  919548668u,  924922419u,
    929051028u, 933179622u,  937308221u,  939994987u,  941567576u,  943205685u,  944516154u,
    945826441u, 947202104u,  948707910u,  950508718u,  952214850u,  953854099u,  955427441u,
    893596928u, 900019402u,  906572942u,  913060941u,  919876608u,  926167908u,  932983460u,
    938422785u, 941896009u,  944517275u,  946614282u,  948645616u,  950939054u,  952773777u,
    954739414u, 956835864u,  959193569u,  961715699u,  964011437u,  966240468u,  968338108u,
    893596928u, 901395679u,  909391032u,  917845130u,  927020111u,  935867408u,  942617477u,
    947204847u, 951923264u,  956445060u,  960377049u,  964177968u,  967454502u,  970599904u,
    973679768u, 977414667u,  980428463u,  983704044u,  986814614u,  989766627u,  992585515u,
    893596928u, 903034112u,  913192187u,  924464373u,  936654052u,  944387272u,  951137439u,
    957101168u, 963458100u,  969618403u,  975909707u,  981152452u,  986722853u,  992555368u,
    998977691u, 1005334369u, 1009003935u, 1011887292u, 1014574035u, 1017129542u, 1019553998u,
    893596928u, 904410400u,  916010301u,  928986456u,  940717417u,  947402097u,  953365866u,
    958281053u, 962409809u,  966341957u,  970929448u,  974927147u,  978662702u,  982201648u,
    985478451u, 988558645u,  991507767u,  994325817u,  997078331u,  999634237u,  1002124607u,
    893596928u, 905721161u,  918566290u,  932132308u,  942028298u,  947009074u,  951072334u,
    954087010u, 956839530u,  959264384u,  961558164u,  963720871u,  965752506u,  967653067u,
    969553628u, 971323116u,  973027067u,  974731018u,  976303897u,  977876775u,  979384116u,
    893596928u, 906311011u,  919680458u,  932722219u,  941504121u,  945370795u,  948254418u,
    950744809u, 952907528u,  954939173u,  956839744u,  958674778u,  960378739u,  962017162u,
    963655585u, 965162935u,  966604748u,  968046560u,  969422836u,  970733571u,  972044300u,
    893596928u, 906573183u,  919877124u,  931804794u,  940062419u,  943142671u,  945370941u,
    947271521u, 948975492u,  950613925u,  952186820u,  953694177u,  955070462u,  956446732u,
    957757465u, 959068198u,  960247858u,  961427518u,  962607177u,  963721300u,  964769887u,
    893596928u, 906311059u,  918763048u,  929838767u,  937375503u,  941045588u,  942880647u,
    944650165u, 946354142u,  947927042u,  949434389u,  950876198u,  952186935u,  953497671u,
    954742870u, 955988069u,  957102196u,  958281857u,  959330447u,  960379036u,  961427625u,
    893596928u, 905721248u,  917059140u,  926627547u,  933115717u,  937703313u,  940390348u,
    941570048u, 942946325u,  944257065u,  945502268u,  946681935u,  947796064u,  948910193u,
    949958785u, 950941840u,  951924895u,  952907950u,  953825468u,  954677450u,  955529431u,
    893596928u, 904345007u,  914241115u,  922760957u,  928593779u,  932460488u,  935671814u,
    938031138u, 940062779u,  941242449u,  942422119u,  943536252u,  944584847u,  945567906u,
    946550965u, 947468486u,  948386007u,  949237991u,  950089975u,  950876423u,  951662870u,
    893596928u, 902772149u,  911160936u,  918370067u,  923547545u,  927283196u,  929970210u,
    931870789u, 934033501u,  936065141u,  937965708u,  939669665u,  940521654u,  941373641u,
    942160093u, 942946543u,  943667457u,  944388370u,  945109283u,  945830195u,  946485571u,
    893596928u, 900937139u,  907753065u,  913717019u,  918304689u,  921581585u,  924006461u,
    925710430u, 927611008u,  929642654u,  931477691u,  933312726u,  935016688u,  936655112u,
    938228000u, 939604278u,  940325196u,  941046113u,  941701494u,  942356874u,  943012253u,
    893596928u, 899560880u,  905066081u,  909850390u,  913520569u,  915879966u,  917649489u,
    919025788u, 920074403u,  920926404u,  922368226u,  923678973u,  924924184u,  926103858u,
    927283530u, 928397666u,  929446264u,  930494862u,  931477923u,  932460984u,  933378508u,
    893596928u, 898053543u,  901920337u,  905393923u,  908146607u,  910047267u,  911489120u,
    912406673u, 913127609u,  913717479u,  914569484u,  915618094u,  916666703u,  917649774u,
    918632843u, 919550375u,  920402370u,  921254365u,  922040822u,  922827278u,  923613734u,
    893596928u, 896480663u,  899364416u,  901658346u,  903296914u,  904411161u,  904869987u,
    904673437u, 904214738u,  903559430u,  902510904u,  901200234u,  900151702u,  898972092u,
    899168734u, 899365375u,  899562015u,  899758653u,  899955291u,  900151927u,  900348563u,
    893596928u, 895169927u,  896546339u,  897726140u,  898381656u,  898381804u,  897792061u,
    897005692u, 895629499u,  893991151u,  892156190u,  890714442u,  889207153u,  887765385u,
    886258090u, 885995976u,  885733862u,  885471746u,  885209630u,  885013048u,  884750930u,
    893596928u, 894186873u,  894645769u,  894776980u,  894580510u,  893925287u,  892680213u,
    891500620u, 889731209u,  887765186u,  885733616u,  883702040u,  881932597u,  880490834u,
    879049066u, 878590338u,  878131610u,  877672880u,  877214150u,  876820955u,  876427760u,
    893596928u, 893269357u,  892745193u,  891958886u,  890910430u,  889468755u,  887699401u,
    886126611u, 884029511u,  881735801u,  879245483u,  877213897u,  875509990u,  873740545u,
    872233240u, 870791471u,  869349701u,  868039002u,  866662767u,  865417603u,  864172438u,
    893596928u, 892548449u,  891172303u,  889599546u,  887764644u,  885864195u,  883505002u,
    881604536u, 879179785u,  877017139u,  874526815u,  872298621u,  868759710u,  865286327u,
    862402766u, 860371173u,  858470651u,  856570128u,  854735140u,  852965688u,  851261771u,
    893596928u, 891696467u,  889468335u,  886978056u,  884356703u,  882063017u,  879179515u,
    876885819u, 874329984u,  871315389u,  866072574u,  861354010u,  856832053u,  852834376u,
    849426525u, 845363315u,  842348680u,  839465117u,  836647089u,  833960133u,  831404247u,
    893596928u, 890844483u,  887633289u,  884422095u,  881145365u,  878327373u,  875247242u,
    872494781u, 867055344u,  862009120u,  856700752u,  851982202u,  846870438u,  842414024u,
    838416358u, 834418692u,  830355475u,  827013155u,  823801907u,  820721730u,  817707088u,
    893596928u, 889599275u,  885405015u,  881145218u,  876950955u,  873608651u,  867710444u,
    861222410u, 855389733u,  849622591u,  844052054u,  838678125u,  834352768u,  829437589u,
    824719014u, 820197049u,  816133835u,  812136153u,  806958831u,  798897926u,  789526299u,
    893596928u, 888747287u,  883766574u,  878851395u,  874198357u,  868365667u,  860435824u,
    853620092u, 847066503u,  840775057u,  834876827u,  829699491u,  824587690u,  819148211u,
    815084984u, 810497471u,  806565316u,  797586890u,  788346319u,  782579156u,  775632345u,
    893596928u, 888026372u,  882259206u,  876885253u,  871314693u,  863188228u,  854799618u,
    847525123u, 840709377u,  833959167u,  827995388u,  821966076u,  817575161u,  812332279u,
    807810294u, 800863476u,  790508785u,  781268210u,  773207279u,  767440107u,  759313644u,
    893596928u, 887239916u,  880948438u,  875115714u,  867185841u,  858338466u,  849360019u,
    842019975u, 834745468u,  828257394u,  822293609u,  816526433u,  811152473u,  806564947u,
    797652044u, 787887169u,  777532469u,  768226354u,  760099882u,  753153062u,  746140706u,
    893596928u, 886846682u,  880161970u,  874001549u,  864302191u,  855061590u,  846017599u,
    838153260u, 830813210u,  824456203u,  818033656u,  812069851u,  806826947u,  798766000u,
    787297169u, 779170689u,  768750441u,  759509840u,  750203712u,  743191344u,  734671646u,
    893596928u, 886650056u,  879965329u,  873215067u,  862860333u,  852767751u,  843133889u,
    834155387u, 826094397u,  819737364u,  813183720u,  807219912u,  798241445u,  786707076u,
    777466475u, 767046217u,  757740072u,  749613583u,  740241909u,  725103062u,  708325815u,
    893596928u, 886781111u,  880227439u,  873411623u,  862729174u,  852505449u,  841888510u,
    832385694u, 823865933u,  815542791u,  808333773u,  800010664u,  787296638u,  776876371u,
    767635770u, 758329624u,  748957942u,  739586259u,  723923116u,  707211397u,  690958431u,
    893596928u, 887239848u,  880882766u,  874394603u,  864891723u,  853947065u,  843788848u,
    834286005u, 824848706u,  815935707u,  808202375u,  799682625u,  784740310u,  771960650u,
    760360656u, 749809272u,  740503037u,  724315528u,  706293014u,  688729255u,  671558715u,
    893596928u, 887764121u,  881865779u,  875574180u,  867250935u,  856240728u,  846016963u,
    836710723u, 827207877u,  818753619u,  811347921u,  803614499u,  789786210u,  777072037u,
    764882160u, 753151042u,  741878580u,  723790319u,  702621876u,  682043142u,  653075635u
};

float xy_to_monotone_hue_angle(float2 xy) {
    // the colours in the chart go clockwise, the angle will go counter clockwise, hence the sign.
    // also we subtract the angle of hue index==0 so the output will start at zero and grow
    // monotonically. this is required for the binary search to work correctly.
    const float2 ill = float2(0.31271, 0.32902); // now adapted to D65, not C (0.31006, 0.31616);
    return fmod(2.0 * MERIAN_PI - 2.52 - atan2(xy.y - ill.y, xy.x - ill.x), 2.0 * MERIAN_PI);
}

float2 munsell_lookup(inout int hue_idx, inout int chroma_idx) {
    hue_idx = ((hue_idx % 40) + 40) % 40;
    chroma_idx = clamp(chroma_idx, 0, 20);
    return unpackHalf2x16ToFloat(munsell_lut[(21 * hue_idx) + chroma_idx]);
}

// compute which side of a line v0--v1 the point p lies
float munsell_side(float2 v0, inout float2 v1, inout float2 p) {
    v1 -= v0;
    p -= v0;
    return (v1.x * p.y) - (v1.y * p.x);
}

// forward method, interpolate same two triangles as the other direction
float2 munsell_to_xy(float2 mhc) {
    float hm = mhc.x * 40.0f;
    float cm = max(mhc.y, 0.0f) * 21.0f;
    int hidxm = int(hm);
    int cidxm = int(cm);
    float hu = hm - float(hidxm);
    float cu = cm - float(cidxm);
    int param = hidxm;
    int param_1 = cidxm + 1;
    float2 _1102 = munsell_lookup(param, param_1);
    float2 res3 = _1102;
    int param_2 = hidxm + 1;
    int param_3 = cidxm + 1;
    float2 _1107 = munsell_lookup(param_2, param_3);
    float2 res2 = _1107;
    int param_4 = hidxm;
    int param_5 = cidxm;
    float2 _1110 = munsell_lookup(param_4, param_5);
    float2 res0 = _1110;
    int param_6 = hidxm + 1;
    int param_7 = cidxm;
    float2 _1114 = munsell_lookup(param_6, param_7);
    float2 res1 = _1114;
    if (hu >= cu) {
        return ((res0 * (1.0f - hu)) + (res1 * (hu - cu))) + (res2 * cu);
    } else {
        return ((res2 * hu) + (res3 * (cu - hu))) + (res0 * (1.0f - cu));
    }
}

// inversion method
// convert xy to polar around illuminant (C)
// find matching munsell box and interpolate the corners
float2 munsell_from_xy(float2 xy) {
    int hidxm = 0;
    int hidxM = 40;
    int cidxm = 0;
    int cidxM = 20;
    float2 param = xy;
    float theta = xy_to_monotone_hue_angle(param);
    float rad2 = dot(xy - float2(0.312709987163543701171875f, 0.3290199935436248779296875f),
                     xy - float2(0.312709987163543701171875f, 0.3290199935436248779296875f));
    for (int i = 0; i < 6; i++) {
        int hidx = (hidxm + hidxM) / 2;
        int cidx = (cidxm + cidxM) / 2;
        int param_1 = hidx;
        int param_2 = cidx;
        float2 _1172 = munsell_lookup(param_1, param_2);
        float2 res = _1172;
        float2 param_3 = res;
        float th = xy_to_monotone_hue_angle(param_3);
        float r2 = dot(res - float2(0.312709987163543701171875f, 0.3290199935436248779296875f),
                       res - float2(0.312709987163543701171875f, 0.3290199935436248779296875f));
        if (th <= theta) {
            hidxm = hidx;
        } else {
            hidxM = hidx;
        }
        if (r2 <= rad2) {
            cidxm = cidx;
        } else {
            cidxM = cidx;
        }
        bool _1199 = hidxM <= (hidxm + 1);
        bool _1206;
        if (_1199) {
            _1206 = cidxM <= (cidxm + 1);
        } else {
            _1206 = _1199;
        }
        if (_1206) {
            break;
        }
    }
    float u2;
    float u0;
    float u1;
    float u3;
    for (int i_1 = 0; i_1 < 6; i_1++) {
        int param_4 = hidxm;
        int param_5 = cidxm + 1;
        float2 _1221 = munsell_lookup(param_4, param_5);
        float2 res3 = _1221;
        int param_6 = hidxm + 1;
        int param_7 = cidxm + 1;
        float2 _1226 = munsell_lookup(param_6, param_7);
        float2 res2 = _1226;
        int param_8 = hidxm;
        int param_9 = cidxm;
        float2 _1229 = munsell_lookup(param_8, param_9);
        float2 res0 = _1229;
        int param_10 = hidxm + 1;
        int param_11 = cidxm;
        float2 _1233 = munsell_lookup(param_10, param_11);
        float2 res1 = _1233;
        float2 param_12 = res0;
        float2 param_13 = res1;
        float2 param_14 = xy;
        float _1236 = munsell_side(param_12, param_13, param_14);
        float s0 = _1236;
        float2 param_15 = res1;
        float2 param_16 = res2;
        float2 param_17 = xy;
        float _1239 = munsell_side(param_15, param_16, param_17);
        float s1 = _1239;
        float2 param_18 = res2;
        float2 param_19 = res3;
        float2 param_20 = xy;
        float _1242 = munsell_side(param_18, param_19, param_20);
        float s2 = _1242;
        float2 param_21 = res3;
        float2 param_22 = res0;
        float2 param_23 = xy;
        float _1245 = munsell_side(param_21, param_22, param_23);
        float s3 = _1245;
        if ((s0 < 0.0f) && (cidxm > 0)) {
            cidxm--;
        } else {
            if ((s0 < 0.0f) && (cidxm == 0)) {
                hidxm = (((hidxm + 20) % 40) + 40) % 40;
            } else {
                if ((s2 < 0.0f) && (cidxm < 19)) {
                    cidxm++;
                }
            }
        }
        if (s1 < 0.0f) {
            hidxm++;
        } else {
            if (s3 < 0.0f) {
                hidxm--;
            }
        }
        bool _1298 = ((s0 >= 0.0f) && (s1 >= 0.0f)) && (s3 >= 0.0f);
        bool _1306;
        if (_1298) {
            _1306 = (s2 >= 0.0f) || (cidxm >= 19);
        } else {
            _1306 = _1298;
        }
        if (_1306) {
            float2 param_24 = res0;
            float2 param_25 = res1;
            float2 param_26 = res2;
            float _1312 = munsell_side(param_24, param_25, param_26);
            float t0 = _1312;
            float2 param_27 = res2;
            float2 param_28 = res3;
            float2 param_29 = res0;
            float _1316 = munsell_side(param_27, param_28, param_29);
            float t1 = _1316;
            bool _1318 = cidxm > 0;
            bool _1326;
            if (_1318) {
                _1326 = (s0 + s1) <= t0;
            } else {
                _1326 = _1318;
            }
            if (_1326) {
                u2 = s0 / t0;
                u0 = s1 / t0;
                u1 = (1.0f - u0) - u2;
                u3 = 0.0f;
            } else {
                u2 = s3 / t1;
                u0 = s2 / t1;
                u3 = (1.0f - u0) - u2;
                u1 = 0.0f;
            }
            float hi = (((u0 * float(hidxm)) + (u1 * (float(hidxm) + 1.0f))) +
                        (u2 * (float(hidxm) + 1.0f))) +
                       (u3 * float(hidxm));
            float ci =
                (((u0 * float(cidxm)) + (u1 * float(cidxm))) + (u2 * (float(cidxm) + 1.0f))) +
                (u3 * (float(cidxm) + 1.0f));
            return float2(hi / 40.0f, max(0.0f, ci / 21.0f));
        }
    }
}

}
