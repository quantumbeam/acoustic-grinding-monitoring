
import os

import re

import pandas as pd

import numpy as np

from collections import defaultdict

from fft_processing import calculate_fft_power

def get_ae_files():
    return [
        'ae_data/exp3/NaCl/3rd/20251217_173239NaCl_grind_for_150um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_172501NaCl_grind_for_150um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_184850NaCl_grind_for_250um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_184633NaCl_grind_for_250um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_184936NaCl_grind_for_250um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_172415NaCl_grind_for_150um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_184718NaCl_grind_for_250um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_173411NaCl_grind_for_150um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_181524NaCl_grind_for_200um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_173108NaCl_grind_for_150um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_184501NaCl_grind_for_250um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_173022NaCl_grind_for_150um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_173800NaCl_grind_for_150um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_172936NaCl_grind_for_150um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_181610NaCl_grind_for_200um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_181439NaCl_grind_for_200um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_172850NaCl_grind_for_150um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_172804NaCl_grind_for_150um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_181349NaCl_grind_for_200um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_181656NaCl_grind_for_200um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_173543NaCl_grind_for_150um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_181914NaCl_grind_for_200um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_172326NaCl_grind_for_150um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_181742NaCl_grind_for_200um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_184412NaCl_grind_for_250um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_181828NaCl_grind_for_200um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_182045NaCl_grind_for_200um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_173325NaCl_grind_for_150um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_172547NaCl_grind_for_150um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_182303NaCl_grind_for_200um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_172718NaCl_grind_for_150um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_173457NaCl_grind_for_150um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_182000NaCl_grind_for_200um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_173154NaCl_grind_for_150um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_184804NaCl_grind_for_250um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_184547NaCl_grind_for_250um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_173629NaCl_grind_for_150um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_173932NaCl_grind_for_150um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_172633NaCl_grind_for_150um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_173714NaCl_grind_for_150um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_182217NaCl_grind_for_200um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_173846NaCl_grind_for_150um_3rd.csv',
        'ae_data/exp3/NaCl/3rd/20251217_182131NaCl_grind_for_200um_3rd.csv',
        'ae_data/exp3/NaCl/1st/20251217_161520NaCl_grind_for_150um.csv',
        'ae_data/exp3/NaCl/1st/20251217_162821NaCl_grind_for_150um.csv',
        'ae_data/exp3/NaCl/1st/20251217_155918NaCl_grind_for_250um.csv',
        'ae_data/exp3/NaCl/1st/20251217_162431NaCl_grind_for_150um.csv',
        'ae_data/exp3/NaCl/1st/20251217_161345NaCl_grind_for_150um.csv',
        'ae_data/exp3/NaCl/1st/20251217_160226NaCl_grind_for_250um.csv',
        'ae_data/exp3/NaCl/1st/20251217_161824NaCl_grind_for_150um.csv',
        'ae_data/exp3/NaCl/1st/20251217_164537NaCl_grind_for_200um.csv',
        'ae_data/exp3/NaCl/1st/20251217_164102NaCl_grind_for_200um.csv',
        'ae_data/exp3/NaCl/1st/20251217_162649NaCl_grind_for_150um.csv',
        'ae_data/exp3/NaCl/1st/20251217_162041NaCl_grind_for_150um.csv',
        'ae_data/exp3/NaCl/1st/20251217_162127NaCl_grind_for_150um.csv',
        'ae_data/exp3/NaCl/1st/20251217_163710NaCl_grind_for_200um.csv',
        'ae_data/exp3/NaCl/1st/20251217_161910NaCl_grind_for_150um.csv',
        'ae_data/exp3/NaCl/1st/20251217_162603NaCl_grind_for_150um.csv',
        'ae_data/exp3/NaCl/1st/20251217_163759NaCl_grind_for_200um.csv',
        'ae_data/exp3/NaCl/1st/20251217_161738NaCl_grind_for_150um.csv',
        'ae_data/exp3/NaCl/1st/20251217_161434NaCl_grind_for_150um.csv',
        'ae_data/exp3/NaCl/1st/20251217_162517NaCl_grind_for_150um.csv',
        'ae_data/exp3/NaCl/1st/20251217_164451NaCl_grind_for_200um.csv',
        'ae_data/exp3/NaCl/1st/20251217_164406NaCl_grind_for_200um.csv',
        'ae_data/exp3/NaCl/1st/20251217_160444NaCl_grind_for_250um.csv',
        'ae_data/exp3/NaCl/1st/20251217_164016NaCl_grind_for_200um.csv',
        'ae_data/exp3/NaCl/1st/20251217_164320NaCl_grind_for_200um.csv',
        'ae_data/exp3/NaCl/1st/20251217_164148NaCl_grind_for_200um.csv',
        'ae_data/exp3/NaCl/1st/20251217_160531NaCl_grind_for_250um.csv',
        'ae_data/exp3/NaCl/1st/20251217_161955NaCl_grind_for_150um.csv',
        'ae_data/exp3/NaCl/1st/20251217_164234NaCl_grind_for_200um.csv',
        'ae_data/exp3/NaCl/1st/20251217_163845NaCl_grind_for_200um.csv',
        'ae_data/exp3/NaCl/1st/20251217_163931NaCl_grind_for_200um.csv',
        'ae_data/exp3/NaCl/1st/20251217_162213NaCl_grind_for_150um.csv',
        'ae_data/exp3/NaCl/1st/20251217_162259NaCl_grind_for_150um.csv',
        'ae_data/exp3/NaCl/1st/20251217_160312NaCl_grind_for_250um.csv',
        'ae_data/exp3/NaCl/1st/20251217_160140NaCl_grind_for_250um.csv',
        'ae_data/exp3/NaCl/1st/20251217_162345NaCl_grind_for_150um.csv',
        'ae_data/exp3/NaCl/1st/20251217_160008NaCl_grind_for_250um.csv',
        'ae_data/exp3/NaCl/1st/20251217_162735NaCl_grind_for_150um.csv',
        'ae_data/exp3/NaCl/1st/20251217_160054NaCl_grind_for_250um.csv',
        'ae_data/exp3/NaCl/1st/20251217_161606NaCl_grind_for_150um.csv',
        'ae_data/exp3/NaCl/1st/20251217_161652NaCl_grind_for_150um.csv',
        'ae_data/exp3/NaCl/1st/20251217_164623NaCl_grind_for_200um.csv',
        'ae_data/exp3/NaCl/1st/20251217_160358NaCl_grind_for_250um.csv',
        'ae_data/exp3/NaCl/2nd/20251217_183505NaCl_grind_for_250um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_180139NaCl_grind_for_200um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_171600NaCl_grind_for_150um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_180400NaCl_grind_for_200um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_181006NaCl_grind_for_200um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_170604NaCl_grind_for_150um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_171646NaCl_grind_for_150um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_183809NaCl_grind_for_250um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_183637NaCl_grind_for_250um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_180617NaCl_grind_for_200um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_183551NaCl_grind_for_250um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_180228NaCl_grind_for_200um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_180446NaCl_grind_for_200um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_171429NaCl_grind_for_150um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_180532NaCl_grind_for_200um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_180314NaCl_grind_for_200um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_170736NaCl_grind_for_150um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_170822NaCl_grind_for_150um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_170650NaCl_grind_for_150um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_171257NaCl_grind_for_150um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_171211NaCl_grind_for_150um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_183723NaCl_grind_for_250um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_183855NaCl_grind_for_250um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_171514NaCl_grind_for_150um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_183940NaCl_grind_for_250um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_170954NaCl_grind_for_150um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_180835NaCl_grind_for_200um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_180703NaCl_grind_for_200um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_180749NaCl_grind_for_200um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_171818NaCl_grind_for_150um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_181052NaCl_grind_for_200um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_180921NaCl_grind_for_200um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_170908NaCl_grind_for_150um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_170125NaCl_grind_for_150um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_170433NaCl_grind_for_150um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_171732NaCl_grind_for_150um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_170519NaCl_grind_for_150um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_170301NaCl_grind_for_150um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_171343NaCl_grind_for_150um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_171125NaCl_grind_for_150um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_170215NaCl_grind_for_150um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_171039NaCl_grind_for_150um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_170346NaCl_grind_for_150um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_183416NaCl_grind_for_250um_2nd.csv',
        'ae_data/exp3/NaCl/2nd/20251217_171904NaCl_grind_for_150um_2nd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_194134Ajinomoto_grind_for100um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_205908Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_204128Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_205650Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_204300Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_191128Ajinomoto_grind_for200um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_191300Ajinomoto_grind_for200um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_205040Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_205822Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_204039Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_194354Ajinomoto_grind_for100um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_204954Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_190911Ajinomoto_grind_for200um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_205604Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_205127Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_194657Ajinomoto_grind_for100um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_205432Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_191042Ajinomoto_grind_for200um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_194915Ajinomoto_grind_for100um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_205736Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_205518Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_204432Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_205954Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_194743Ajinomoto_grind_for100um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_205346Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_194440Ajinomoto_grind_for100um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_194525Ajinomoto_grind_for100um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_190825Ajinomoto_grind_for200um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_194044Ajinomoto_grind_for100um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_204737Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_205301Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_204822Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_190736Ajinomoto_grind_for200um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_194307Ajinomoto_grind_for100um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_205214Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_210040Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_204651Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_204908Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_194829Ajinomoto_grind_for100um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_204346Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_194611Ajinomoto_grind_for100um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_204214Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_204605Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_191214Ajinomoto_grind_for200um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_210126Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_194221Ajinomoto_grind_for100um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_190957Ajinomoto_grind_for200um_3rd.csv',
        'ae_data/exp3/Ajinomoto/3rd/20251219_204518Ajinomoto_grind_for50um_3rd.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_191851Ajinomoto_grind_for100um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_201312Ajinomoto_grind_for50um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_200618Ajinomoto_grind_for50um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_191720Ajinomoto_grind_for100um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_185327Ajinomoto_grind_for200um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_185242Ajinomoto_grind_for200um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_200400Ajinomoto_grind_for50um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_192155Ajinomoto_grind_for100um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_195839Ajinomoto_grind_for50um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_185459Ajinomoto_grind_for200um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_201139Ajinomoto_grind_for50um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_200921Ajinomoto_grind_for50um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_184938Ajinomoto_grind_for200um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_195925Ajinomoto_grind_for50um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_192109Ajinomoto_grind_for100um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_200057Ajinomoto_grind_for50um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_185413Ajinomoto_grind_for200um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_200532Ajinomoto_grind_for50um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_201445Ajinomoto_grind_for50um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_185156Ajinomoto_grind_for200um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_200446Ajinomoto_grind_for50um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_200749Ajinomoto_grind_for50um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_192327Ajinomoto_grind_for100um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_195618Ajinomoto_grind_for50um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_192023Ajinomoto_grind_for100um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_200704Ajinomoto_grind_for50um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_195753Ajinomoto_grind_for50um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_201358Ajinomoto_grind_for50um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_200314Ajinomoto_grind_for50um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_201225Ajinomoto_grind_for50um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_200142Ajinomoto_grind_for50um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_191805Ajinomoto_grind_for100um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_192241Ajinomoto_grind_for100um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_200011Ajinomoto_grind_for50um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_192458Ajinomoto_grind_for100um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_191631Ajinomoto_grind_for100um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_192412Ajinomoto_grind_for100um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_201053Ajinomoto_grind_for50um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_185110Ajinomoto_grind_for200um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_191937Ajinomoto_grind_for100um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_195707Ajinomoto_grind_for50um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_185024Ajinomoto_grind_for200um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_200228Ajinomoto_grind_for50um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_201007Ajinomoto_grind_for50um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_184849Ajinomoto_grind_for200um.csv',
        'ae_data/exp3/Ajinomoto/1st/20251219_200835Ajinomoto_grind_for50um.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_202634Ajinomoto_grind_for50um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_202110Ajinomoto_grind_for50um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_202156Ajinomoto_grind_for50um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_203242Ajinomoto_grind_for50um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_190125Ajinomoto_grind_for200um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_202852Ajinomoto_grind_for50um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_202806Ajinomoto_grind_for50um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_193517Ajinomoto_grind_for100um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_203024Ajinomoto_grind_for50um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_203156Ajinomoto_grind_for50um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_193125Ajinomoto_grind_for100um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_203414Ajinomoto_grind_for50um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_193211Ajinomoto_grind_for100um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_203632Ajinomoto_grind_for50um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_202243Ajinomoto_grind_for50um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_193257Ajinomoto_grind_for100um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_185953Ajinomoto_grind_for200um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_202329Ajinomoto_grind_for50um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_193603Ajinomoto_grind_for100um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_192907Ajinomoto_grind_for100um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_203546Ajinomoto_grind_for50um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_202720Ajinomoto_grind_for50um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_192953Ajinomoto_grind_for100um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_203328Ajinomoto_grind_for50um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_190039Ajinomoto_grind_for200um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_202023Ajinomoto_grind_for50um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_203110Ajinomoto_grind_for50um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_201936Ajinomoto_grind_for50um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_193039Ajinomoto_grind_for100um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_190428Ajinomoto_grind_for200um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_201800Ajinomoto_grind_for50um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_190342Ajinomoto_grind_for200um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_193430Ajinomoto_grind_for100um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_202416Ajinomoto_grind_for50um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_193649Ajinomoto_grind_for100um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_185904Ajinomoto_grind_for200um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_202502Ajinomoto_grind_for50um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_190210Ajinomoto_grind_for200um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_203500Ajinomoto_grind_for50um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_190256Ajinomoto_grind_for200um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_201850Ajinomoto_grind_for50um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_202548Ajinomoto_grind_for50um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_193344Ajinomoto_grind_for100um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_202938Ajinomoto_grind_for50um_2nd.csv',
        'ae_data/exp3/Ajinomoto/2nd/20251219_192818Ajinomoto_grind_for100um_2nd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_185626Citricacid_grind25min_for50um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_200601Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_201254Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_195432Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_190319Citricacid_grind25min_for50um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_181121Citricacid_grind25min_for100um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_200343Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_200951Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_200515Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_185712Citricacid_grind25min_for50um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_185758Citricacid_grind25min_for50um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_195821Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_195954Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_181035Citricacid_grind25min_for100um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_195518Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_181252Citricacid_grind25min_for100um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_195604Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_190233Citricacid_grind25min_for50um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_200429Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_200819Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_190147Citricacid_grind25min_for50um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_190102Citricacid_grind25min_for50um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_195343Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_181206Citricacid_grind25min_for100um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_185234Citricacid_grind25min_for50um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_181424Citricacid_grind25min_for100um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_195650Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_200733Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_185323Citricacid_grind25min_for50um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_185409Citricacid_grind25min_for50um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_195736Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_200257Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_180900Citricacid_grind25min_for100um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_200039Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_201123Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_200905Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_185540Citricacid_grind25min_for50um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_185455Citricacid_grind25min_for50um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_200125Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_185930Citricacid_grind25min_for50um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_180949Citricacid_grind25min_for100um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_200647Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_201340Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_201208Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_185844Citricacid_grind25min_for50um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_181338Citricacid_grind25min_for100um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_201037Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_200211Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_190016Citricacid_grind25min_for50um_3rd.csv',
        'ae_data/exp3/Citricacid/3rd/20251218_195907Citricacid_grind25min_for20um_3rd.csv',
        'ae_data/exp3/Citricacid/1st/20251218_192514Citricacid_grind25min_for20um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_191953Citricacid_grind25min_for20um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_182408Citricacid_grind25min_for50um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_182455Citricacid_grind25min_for50um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_182626Citricacid_grind25min_for50um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_191432Citricacid_grind25min_for20um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_182019Citricacid_grind25min_for50um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_175633Citricacid_grind25min_for100um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_175415Citricacid_grind25min_for100um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_182844Citricacid_grind25min_for50um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_192256Citricacid_grind25min_for20um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_181930Citricacid_grind25min_for50um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_192210Citricacid_grind25min_for20um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_191603Citricacid_grind25min_for20um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_190910Citricacid_grind25min_for20um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_182540Citricacid_grind25min_for50um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_192342Citricacid_grind25min_for20um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_191042Citricacid_grind25min_for20um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_191649Citricacid_grind25min_for20um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_192428Citricacid_grind25min_for20um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_191346Citricacid_grind25min_for20um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_191735Citricacid_grind25min_for20um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_182237Citricacid_grind25min_for50um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_182929Citricacid_grind25min_for50um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_192125Citricacid_grind25min_for20um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_175158Citricacid_grind25min_for100um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_175547Citricacid_grind25min_for100um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_191300Citricacid_grind25min_for20um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_191517Citricacid_grind25min_for20um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_191214Citricacid_grind25min_for20um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_191821Citricacid_grind25min_for20um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_175501Citricacid_grind25min_for100um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_191907Citricacid_grind25min_for20um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_190735Citricacid_grind25min_for20um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_182105Citricacid_grind25min_for50um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_190824Citricacid_grind25min_for20um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_190956Citricacid_grind25min_for20um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_182758Citricacid_grind25min_for50um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_182712Citricacid_grind25min_for50um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_175329Citricacid_grind25min_for100um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_175109Citricacid_grind25min_for100um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_191128Citricacid_grind25min_for20um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_182323Citricacid_grind25min_for50um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_175244Citricacid_grind25min_for100um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_182151Citricacid_grind25min_for50um.csv',
        'ae_data/exp3/Citricacid/1st/20251218_192039Citricacid_grind25min_for20um.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_193813Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_193033Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_194506Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_194724Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_184509Citricacid_grind25min_for50um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_180355Citricacid_grind25min_for100um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_193641Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_184812Citricacid_grind25min_for50um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_194030Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_193205Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_184858Citricacid_grind25min_for50um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_194638Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_180310Citricacid_grind25min_for100um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_193944Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_194116Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_183902Citricacid_grind25min_for50um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_194552Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_183948Citricacid_grind25min_for50um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_193858Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_184251Citricacid_grind25min_for50um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_193727Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_180527Citricacid_grind25min_for100um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_193119Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_183730Citricacid_grind25min_for50um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_192858Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_184205Citricacid_grind25min_for50um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_194202Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_193509Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_194856Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_192947Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_194942Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_183641Citricacid_grind25min_for50um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_184555Citricacid_grind25min_for50um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_180052Citricacid_grind25min_for100um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_180138Citricacid_grind25min_for100um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_180003Citricacid_grind25min_for100um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_180224Citricacid_grind25min_for100um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_180441Citricacid_grind25min_for100um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_193251Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_183816Citricacid_grind25min_for50um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_184640Citricacid_grind25min_for50um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_184423Citricacid_grind25min_for50um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_194810Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_184119Citricacid_grind25min_for50um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_194420Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_184726Citricacid_grind25min_for50um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_184034Citricacid_grind25min_for50um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_194248Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_194334Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_193423Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_184337Citricacid_grind25min_for50um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_195027Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_193555Citricacid_grind25min_for20um_2nd.csv',
        'ae_data/exp3/Citricacid/2nd/20251218_193337Citricacid_grind25min_for20um_2nd.csv',
    ]

def get_powder_files():
    return [
        'powder_size_distribution_data/exp3/NaCl/3rd/20251217_174127_NaCl_grind_for150um_3rd.csv',
        'powder_size_distribution_data/exp3/NaCl/3rd/20251217_185100_NaCl_grind_for250um_3rd.csv',
        'powder_size_distribution_data/exp3/NaCl/3rd/20251217_182914_NaCl_grind_for200um_3rd.csv',
        'powder_size_distribution_data/exp3/NaCl/1st/20251217_165226_NaCl_grind_for200um.csv',
        'powder_size_distribution_data/exp3/NaCl/1st/20251217_161354_NaCl_grind_for250um.csv',
        'powder_size_distribution_data/exp3/NaCl/1st/20251217_163133_NaCl_grind_for150um.csv',
        'powder_size_distribution_data/exp3/NaCl/2nd/20251217_181405_NaCl_grind_for200um_2nd.csv',
        'powder_size_distribution_data/exp3/NaCl/2nd/20251217_184133_NaCl_grind_for250um_2nd.csv',
        'powder_size_distribution_data/exp3/NaCl/2nd/20251217_172511_NaCl_grind_for150um_2nd.csv',
        'powder_size_distribution_data/exp3/Ajinomoto/3rd/20251219_210418_Ajinomoto_grind_for50um_3rd.csv',
        'powder_size_distribution_data/exp3/Ajinomoto/3rd/20251219_195649_Ajinomoto_grind_for100um_3rd.csv',
        'powder_size_distribution_data/exp3/Ajinomoto/3rd/20251219_191641_Ajinomoto_grind_for200um_3rd.csv',
        'powder_size_distribution_data/exp3/Ajinomoto/1st/20251219_201818_Ajinomoto_grind_for50um.csv',
        'powder_size_distribution_data/exp3/Ajinomoto/1st/20251219_190013_Ajinomoto_grind_for200um.csv',
        'powder_size_distribution_data/exp3/Ajinomoto/1st/20251219_192939_Ajinomoto_grind_for100um.csv',
        'powder_size_distribution_data/exp3/Ajinomoto/2nd/20251219_204111_Ajinomoto_grind_for50um_2nd.csv',
        'powder_size_distribution_data/exp3/Ajinomoto/2nd/20251219_190747_Ajinomoto_grind_for200um_2nd.csv',
        'powder_size_distribution_data/exp3/Ajinomoto/2nd/20251219_194205_Ajinomoto_grind_for100um_2nd.csv',
        'powder_size_distribution_data/exp3/Citricacid/3rd/20251218_201543_Citricacid_grind_for20um_3rd.csv',
        'powder_size_distribution_data/exp3/Citricacid/3rd/20251218_190854_Citricacid_grind_for50um_3rd.csv',
        'powder_size_distribution_data/exp3/Citricacid/3rd/20251218_182002_Citricacid_grind_for100um_3rd.csv',
        'powder_size_distribution_data/exp3/Citricacid/1st/20251218_183723_Citricacid_grind_for50um.csv',
        'powder_size_distribution_data/exp3/Citricacid/1st/20251218_180031_Citricacid_grind_for100um.csv',
        'powder_size_distribution_data/exp3/Citricacid/1st/20251218_192938_Citricacid_grind_for20um.csv',
        'powder_size_distribution_data/exp3/Citricacid/2nd/20251218_181015_Citricacid_grind_for100um_2nd.csv',
        'powder_size_distribution_data/exp3/Citricacid/2nd/20251218_185354_Citricacid_grind_for50um_2nd.csv',
        'powder_size_distribution_data/exp3/Citricacid/2nd/20251218_195416_Citricacid_grind_for20um_2nd.csv',
    ]

def parse_filename(filename):
    timestamp = None
    material = None
    target_size = None

    # Pattern 1: TIMESTAMP_MATERIAL_GRIND_TYPE_SIZEum (e.g., powder files, some AE files)
    match1 = re.search(r'(\d{8}_\d{6})_([a-zA-Z]+)_(grind_for|grind25min_for)(\d+)um', filename)
    if match1:
        timestamp, material_raw, grind_type_part, target_size = match1.groups()
        material = material_raw
    else:
        # Pattern 2: TIMESTAMP_MATERIALGRIND_TYPE_SIZEum (e.g., some AE files like NaCl)
        match2 = re.search(r'(\d{8}_\d{6})([a-zA-Z]+)(?:_grind_for|_grind25min_for)(\d+)um', filename)
        if match2:
            timestamp, material_raw, target_size = match2.groups()
            material = material_raw
        else:
            return None, None, None, None

    trial = '1st'
    if '2nd' in filename:
        trial = '2nd'
    elif '3rd' in filename:
        trial = '3rd'
        
    # Standardize material name for CitricAcid
    if material == 'Citricacid':
        material = 'CitricAcid'

    return material, int(target_size), trial, timestamp

def get_measured_value(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('Dx (50),'):
                return float(line.split(',')[1])
    return None

def generate_raw_data_map():
    powder_files = get_powder_files()
    ae_files = get_ae_files()

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for f in powder_files:
        material, target_size, trial, _ = parse_filename(f)
        if material:
            measured_val = get_measured_value(os.path.abspath(f))
            data[material][target_size][trial]['measured'] = measured_val
    
    latest_ae_files = {}
    for f in ae_files:
        material, target_size, trial, timestamp = parse_filename(f)
        if material:
            key = (material, target_size, trial)
            if key not in latest_ae_files or timestamp > latest_ae_files[key][0]:
                latest_ae_files[key] = (timestamp, f)

    for key, (_, f) in latest_ae_files.items():
        material, target_size, trial = key
        power = calculate_fft_power(os.path.abspath(f))
        print(f"Processing {f}, power: {power}")
        data[material][target_size][trial]['ae_power'] = power

    # Reformat data to match the desired structure
    raw_data_map = {}
    for material, targets in data.items():
        # Correcting material name from 'Citricacid' to 'CitricAcid'
        if material == 'Citricacid':
            material = 'CitricAcid'

        raw_data_map[material] = {'targets': [], 'measured': [], 'ae_power': []}
        
        sorted_targets = sorted(targets.keys(), reverse=True)
        raw_data_map[material]['targets'] = sorted_targets
        
        for target in sorted_targets:
            measured_trials = []
            power_trials = []
            for trial in ['1st', '2nd', '3rd']:
                if trial in targets[target]:
                    measured_trials.append(targets[target][trial].get('measured'))
                    power_trials.append(targets[target][trial].get('ae_power'))

            raw_data_map[material]['measured'].append(measured_trials)
            raw_data_map[material]['ae_power'].append(power_trials)
            
    return raw_data_map

if __name__ == '__main__':
    raw_data_map = generate_raw_data_map()
    import pprint
    pprint.pprint(raw_data_map)
