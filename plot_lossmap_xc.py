import os
import re
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

TOT_INJ_ENERGY =  17.5e4 # 1% of the total energy 17.5e6 # J  
NB = 11220
KB = 2.176E11
BEAM_ENERGY = 45.6e9
EV_TO_JOULE = 1.60218e-19

FCC_EE_WARM_REGIONS_V23 = np.array([
    [0.0, 2.2002250210956813], [2.900225021095681, 2.9802250210956807], 
    [4.230225021095681, 4.310225021095681], [5.560225021095681, 5.860225021095681], 
    [7.110225021095681, 7.1902250210956815], [8.440225021095682, 19.021636933288438], 
    [21.921636933288436, 24.095566014890213], [26.995566014890212, 83.9467184323269], 
    [86.84671843232691, 94.10927511652588], [97.00927511652588, 122.25505466852684], 
    [125.15505466852684, 125.45505466852684], [125.68005466852684, 125.68005466852685], 
    [125.90505466852684, 280.1903764070126], [280.4153764070126, 280.4153764070126], 
    [280.64037640701264, 22011.935992171686], [22012.160992171684, 22012.160992171684], 
    [22012.385992171683, 22225.188674086643], [22225.41367408664, 22225.41367408664], 
    [22225.63867408664, 22225.93867408664], [22228.83867408664, 22243.586622081384], 
    [22246.486622081386, 22335.403176079093], [22338.303176079095, 22467.336337002067], 
    [22470.23633700207, 22547.19338558328], [22550.09338558328, 22656.24610479999], 
    [22657.49610479999, 22657.57610479999], [22658.82610479999, 22659.126104799994], 
    [22660.376104799994, 22660.456104799996], [22661.706104799996, 22661.786104799998], 
    [22662.4861048, 22666.8865548422], [22667.5865548422, 22667.66655484221], 
    [22668.91655484221, 22668.996554842208], [22670.246554842208, 22670.546554842207], 
    [22671.796554842207, 22671.87655484221], [22673.12655484221, 22683.707966754406], 
    [22686.607966754407, 22688.781895836008], [22691.68189583601, 22748.63304825344], 
    [22751.53304825344, 22758.795604937637], [22761.69560493764, 22786.941384489633], 
    [22789.841384489635, 22790.141384489634], [22790.366384489633, 22790.366384489633], 
    [22790.59138448963, 22944.876706228104], [22945.101706228103, 22945.101706228106], 
    [22945.326706228105, 44676.622330340004], [44676.84733034, 44676.847330339995], 
    [44677.072330339994, 44889.87501225503], [44890.100012255025, 44890.100012255025], 
    [44890.32501225502, 44890.625012255034], [44893.525012255035, 44908.27296024977], 
    [44911.17296024977, 45000.089514247484], [45002.989514247485, 45132.02267517048], 
    [45134.922675170485, 45211.8797237517], [45214.7797237517, 45320.932442968406], 
    [45322.182442968406, 45322.26244296841], [45323.51244296841, 45323.8124429684], 
    [45325.0624429684, 45325.1424429684], [45326.3924429684, 45326.47244296841], 
    [45327.1724429684, 45331.57289301058], [45332.27289301058, 45332.35289301058], 
    [45333.60289301058, 45333.68289301059], [45334.93289301059, 45335.232893010594], 
    [45336.482893010594, 45336.5628930106], [45337.8128930106, 45348.394304922804], 
    [45351.294304922805, 45353.46823400441], [45356.36823400441, 45413.31938642187], 
    [45416.21938642187, 45423.481943106075], [45426.38194310608, 45451.62772265809], 
    [45454.527722658095, 45454.8277226581], [45455.052722658096, 45455.052722658096], 
    [45455.277722658095, 45609.56304439665], [45609.78804439665, 45609.78804439665], 
    [45610.01304439665, 67341.30866016546], [67341.53366016547, 67341.53366016547], 
    [67341.75866016548, 67554.56134208044], [67554.78634208045, 67554.78634208045], 
    [67555.01134208046, 67555.31134208046], [67558.21134208045, 67572.9592900752], 
    [67575.8592900752, 67664.77584407291], [67667.6758440729, 67796.70900499586], 
    [67799.60900499586, 67876.5660535771], [67879.4660535771, 67985.6187727938], 
    [67986.8687727938, 67986.9487727938], [67988.1987727938, 67988.49877279381], 
    [67989.74877279381, 67989.82877279382], [67991.07877279382, 67991.15877279382], 
    [67991.85877279381, 67996.25922283597], [67996.95922283597, 67997.03922283597], 
    [67998.28922283597, 67998.36922283597], [67999.61922283597, 67999.91922283598], 
    [68001.16922283598, 68001.24922283598], [68002.49922283598, 68013.08063474817], 
    [68015.98063474816, 68018.15456382977], [68021.05456382976, 68078.00571624722], 
    [68080.90571624722, 68088.16827293142], [68091.06827293141, 68116.31405248342], 
    [68119.21405248341, 68119.51405248341], [68119.73905248342, 68119.73905248342], 
    [68119.96405248342, 68274.24937422191], [68274.47437422191, 68274.47437422191], 
    [68274.69937422192, 90005.9949899894], [90006.2199899894, 90006.21998998942], 
    [90006.44498998942, 90219.24767190438], [90219.47267190438, 90219.47267190438], 
    [90219.69767190439, 90219.99767190439], [90222.89767190439, 90237.64561989912], 
    [90240.54561989912, 90329.46217389684], [90332.36217389684, 90461.39533481981], 
    [90464.2953348198, 90541.25238340105], [90544.15238340104, 90650.30510261773], 
    [90651.55510261773, 90651.63510261773], [90652.88510261773, 90653.18510261775], 
    [90654.43510261775, 90654.51510261776], [90655.76510261776, 90655.84510261775]])

FCC_EE_WARM_REGIONS_V24 = np.array([[0.0, 2.200225021099322], [2.900225021099322, 2.9802250210923376], [4.230225021092338, 4.310225021094084], [5.560225021094084, 5.860225021096994], [7.110225021096994, 7.19022502109874], [8.44022502109874, 35.71083996460073], [38.41083996460073, 38.8938503086887], [41.5938503086887, 97.99168302970648], [100.69168302970648, 106.64570040097752], [109.34570040097752, 138.60250754266457], [141.30250754266456, 141.6025075426762], [141.8275075426762, 141.8275075426762], [142.05250754267618, 297.4744831588002], [297.6994831588002, 297.6994831588002], [297.92448315880023, 21989.80775571188], [21990.03275571188, 21990.03275571188], [21990.25775571188, 22204.590787030353], [22204.81578703035, 22204.81578703035], [22205.04078703035, 22205.34078703035], [22208.04078703035, 22234.50115599489], [22237.20115599489, 22343.067324241438], [22345.76732424144, 22475.1515061424], [22477.8515061424, 22580.385601309143], [22583.085601309143, 22656.186206056605], [22657.436206056605, 22657.51620605662], [22658.76620605662, 22659.066206056614], [22660.316206056614, 22660.396206056612], [22661.646206056612, 22661.726206056625], [22662.426206056625, 22666.82665609883], [22667.52665609883, 22667.606656098826], [22668.856656098826, 22668.936656098827], [22670.186656098827, 22670.48665609883], [22671.73665609883, 22671.816656098832], [22673.066656098832, 22700.337271042335], [22703.037271042336, 22703.52028138643], [22706.22028138643, 22762.618114107434], [22765.318114107435, 22771.272131478723], [22773.972131478724, 22803.228938620414], [22805.928938620415, 22806.228938620417], [22806.453938620416, 22806.453938620412], [22806.67893862041, 22962.100914236562], [22962.32591423656, 22962.325914236557], [22962.550914236555, 44654.434186791164], [44654.65918679116, 44654.659186791156], [44654.884186791154, 44869.217218109574], [44869.44221810957, 44869.44221810958], [44869.66721810958, 44869.96721810958], [44872.66721810958, 44899.127587074116], [44901.82758707411, 45007.69375532066], [45010.393755320656, 45139.77793722162], [45142.477937221614, 45245.01203238836], [45247.71203238836, 45320.81263713579], [45322.06263713579, 45322.14263713578], [45323.39263713578, 45323.69263713578], [45324.94263713578, 45325.022637135786], [45326.272637135786, 45326.35263713578], [45327.05263713578, 45331.45308717797], [45332.15308717797, 45332.23308717797], [45333.48308717797, 45333.563087177965], [45334.813087177965, 45335.113087177975], [45336.363087177975, 45336.44308717798], [45337.69308717798, 45364.96370212147], [45367.66370212147, 45368.14671246557], [45370.846712465565, 45427.24454518658], [45429.944545186576, 45435.898562557864], [45438.59856255786, 45467.855369699566], [45470.55536969956, 45470.855369699544], [45471.08036969954, 45471.08036969954], [45471.30536969954, 45626.72734531565], [45626.95234531565, 45626.95234531565], [45627.17734531565, 67319.06061787016], [67319.28561787016, 67319.28561787016], [67319.51061787017, 67533.84364918858], [67534.06864918859, 67534.06864918859], [67534.29364918859, 67534.59364918858], [67537.29364918858, 67563.75401815311], [67566.4540181531, 67672.32018639962], [67675.02018639962, 67804.40436830054], [67807.10436830054, 67909.63846346727], [67912.33846346727, 67985.43906821472], [67986.68906821472, 67986.76906821472], [67988.01906821472, 67988.31906821474], [67989.56906821474, 67989.64906821474], [67990.89906821474, 67990.97906821474], [67991.67906821474, 67996.0795182569], [67996.7795182569, 67996.8595182569], [67998.1095182569, 67998.1895182569], [67999.4395182569, 67999.7395182569], [68000.9895182569, 68001.06951825689], [68002.31951825689, 68029.59013320039], [68032.29013320038, 68032.77314354447], [68035.47314354447, 68091.87097626545], [68094.57097626544, 68100.52499363669], [68103.22499363669, 68132.48180077839], [68135.18180077839, 68135.48180077838], [68135.70680077838, 68135.70680077838], [68135.93180077839, 68291.35377639449], [68291.5787763945, 68291.5787763945], [68291.8037763945, 89983.68704893017], [89983.91204893017, 89983.91204893017], [89984.13704893018, 90198.4700802486], [90198.69508024861, 90198.6950802486], [90198.9200802486, 90199.22008024857], [90201.92008024857, 90228.38044921312], [90231.08044921311, 90336.94661745962], [90339.64661745961, 90469.03079936057], [90471.73079936056, 90574.2648945273], [90576.9648945273, 90650.06549927473], [90651.31549927473, 90651.39549927472], [90652.64549927472, 90652.94549927475], [90654.19549927475, 90654.27549927475], [90655.52549927475, 90655.60549927475]])

FCC_EE_WARM_REGIONS_V25 = np.array([[0.0, 2.200225021099322], [2.900225021099322, 2.9802250210923376], [4.230225021092338, 4.310225021094084], [5.560225021094084, 5.860225021096994], [7.110225021096994, 7.19022502109874], [8.44022502109874, 35.71083996460073], [38.41083996460073, 38.8938503086887], [41.5938503086887, 97.99168302970648], [100.69168302970648, 106.64570040097752], [109.34570040097752, 138.60250754266457], [141.30250754266456, 141.6025075426762], [141.8275075426762, 141.8275075426762], [142.05250754267618, 297.4744831588002], [297.6994831588002, 297.6994831588002], [297.92448315880023, 21989.808585489973], [21990.03358548997, 21990.033585489975], [21990.258585489973, 22204.591616808433], [22204.81661680843, 22204.81661680843], [22205.04161680843, 22205.34161680843], [22208.04161680843, 22234.50198577296], [22237.20198577296, 22343.06815401952], [22345.76815401952, 22475.15233592048], [22477.85233592048, 22580.386431087223], [22583.086431087224, 22656.187035834682], [22657.437035834682, 22657.517035834688], [22658.767035834688, 22659.06703583469], [22660.31703583469, 22660.397035834692], [22661.647035834692, 22661.727035834705], [22662.427035834706, 22666.827485876896], [22667.527485876897, 22667.607485876903], [22668.857485876903, 22668.937485876908], [22670.187485876908, 22670.48748587691], [22671.73748587691, 22671.817485876913], [22673.067485876913, 22700.338100820405], [22703.038100820406, 22703.52111116451], [22706.221111164512, 22762.618943885514], [22765.318943885515, 22771.272961256785], [22773.972961256786, 22803.229768398494], [22805.929768398495, 22806.229768398498], [22806.454768398497, 22806.454768398493], [22806.67976839849, 22962.101744014642], [22962.32674401464, 22962.326744014637], [22962.551744014636, 44654.43584634746], [44654.660846347455, 44654.66084634745], [44654.885846347446, 44869.21887766588], [44869.44387766588, 44869.44387766588], [44869.66887766588, 44869.96887766588], [44872.66887766588, 44899.12924663041], [44901.829246630405, 45007.695414876966], [45010.39541487696, 45139.77959677792], [45142.479596777914, 45245.01369194466], [45247.71369194466, 45320.814296692086], [45322.064296692086, 45322.14429669209], [45323.39429669209, 45323.69429669208], [45324.94429669208, 45325.024296692085], [45326.274296692085, 45326.35429669208], [45327.05429669208, 45331.45474673426], [45332.15474673426, 45332.23474673426], [45333.48474673426, 45333.56474673427], [45334.81474673427, 45335.114746734274], [45336.364746734274, 45336.44474673428], [45337.69474673428, 45364.96536167777], [45367.66536167777, 45368.14837202186], [45370.84837202186, 45427.246204742885], [45429.94620474288, 45435.90022211417], [45438.60022211417, 45467.85702925586], [45470.557029255855, 45470.85702925585], [45471.08202925585, 45471.08202925585], [45471.30702925585, 45626.72900487195], [45626.95400487195, 45626.95400487194], [45627.17900487194, 67319.06310720468], [67319.28810720469, 67319.28810720469], [67319.5131072047, 67533.8461385231], [67534.07113852311, 67534.07113852311], [67534.29613852312, 67534.5961385231], [67537.2961385231, 67563.75650748763], [67566.45650748763, 67672.32267573415], [67675.02267573414, 67804.40685763507], [67807.10685763507, 67909.6409528018], [67912.3409528018, 67985.44155754925], [67986.69155754925, 67986.77155754925], [67988.02155754925, 67988.32155754926], [67989.57155754926, 67989.65155754927], [67990.90155754927, 67990.98155754927], [67991.68155754927, 67996.08200759142], [67996.78200759142, 67996.86200759142], [67998.11200759142, 67998.19200759142], [67999.44200759142, 67999.74200759143], [68000.99200759143, 68001.07200759143], [68002.32200759143, 68029.59262253491], [68032.29262253491, 68032.775632879], [68035.475632879, 68091.87346559997], [68094.57346559997, 68100.52748297121], [68103.22748297121, 68132.48429011292], [68135.18429011291, 68135.4842901129], [68135.7092901129, 68135.7092901129], [68135.93429011291, 68291.35626572902], [68291.58126572902, 68291.58126572902], [68291.80626572903, 89983.69036804326], [89983.91536804326, 89983.91536804326], [89984.14036804327, 90198.4733993617], [90198.6983993617, 90198.69839936167], [90198.92339936168, 90199.2233993617], [90201.9233993617, 90228.38376832621], [90231.0837683262, 90336.94993657272], [90339.64993657272, 90469.03411847363], [90471.73411847363, 90574.26821364036], [90576.96821364036, 90650.06881838782], [90651.31881838782, 90651.39881838782], [90652.64881838782, 90652.94881838784], [90654.19881838784, 90654.27881838784], [90655.52881838784, 90655.60881838785]])

def check_warm_loss(s, warm_regions):
    return np.any((warm_regions.T[0] < s) & (warm_regions.T[1] > s))

def load_multiple_lossmaps(base_dir, output_dir, single_file=None):
    '''
    Function to produce datframe to handles better the datasets.
    '''
    merged_data = {
        'collimator': {'s': [], 'name': [], 'length': [], 'n': []},
        'aperture': {'s': [], 'name': [], 'length': [], 'n': []}
    }
    missing_files = 0
    from_part = False
        # Check if a single file is provided
    if single_file:
        file_path = os.path.join(base_dir,single_file)
        if os.path.exists(file_path):
            df = pd.read_json(file_path)
            if 'collimator' in df.columns and 'aperture' in df.columns:
                # Safely merge the lists
                
                for col in ['collimator', 'aperture']:
                    for row in ['s', 'name', 'length', 'n']:
                        data = df.at[row, col]
                        if isinstance(data, list):
                            merged_data[col][row].extend(data)
                        else:
                            merged_data[col][row].append(data)
            else:
                print(f"Columns 'collimator' or 'aperture' missing in {file_path}")
                from_part = True
                merged_data['collimator']['length'] = df['coll_end'] - df['coll_start']
                merged_data['collimator']['s'] = df['coll_start'] + (merged_data['collimator']['length'] / 2)
                merged_data['collimator']['name'] = df['coll_name'].tolist()
                merged_data['collimator']['n'] = df['coll_loss'].fillna(0).tolist()
                merged_data['aperture']['n'] = df['aper_loss'].fillna(0).tolist()

        else:
            print(f"File does not exist: {file_path}")
    else:
        # Loop through directories Job.0 to Job.99
        for i in range(100):  # Adjust as needed for your range of jobs
            job_dir = f"Job.{i}/plots"
            file_path = os.path.join(base_dir, job_dir, 'lossmap.json')

            if os.path.exists(file_path):
                df = pd.read_json(file_path)
                if 'collimator' in df.columns and 'aperture' in df.columns:
                    # Safely merge the lists
                    for col in ['collimator', 'aperture']:
                        for row in ['s', 'name', 'length', 'n']:
                            data = df.at[row, col]
                            if isinstance(data, list):
                                merged_data[col][row].extend(data)
                            else:
                                merged_data[col][row].append(data)
                else:
                    print(f"Columns 'collimator' or 'aperture' missing in {file_path}")
            else:
                print(f"Warning: {file_path} does not exist.")
                missing_files += 1
        
        save_lossmap_to_json(merged_data, os.path.join(output_dir,'merged_lossmap.json'))
        print(missing_files)

    collimator_df = pd.DataFrame({
    's': merged_data['collimator']['s'],
    'name': merged_data['collimator']['name'],
    'length': merged_data['collimator']['length'],
    'n': merged_data['collimator']['n'],
    'type': 'collimator'  # Label to distinguish between types
    })  

    if not from_part:

        aperture_df = pd.DataFrame({
            's': merged_data['aperture']['s'],
            'name': merged_data['aperture']['name'],
            #'length': merged_data['aperture']['length'],
            'n': merged_data['aperture']['n'],
            'type': 'aperture'  # Label to distinguish between types
        })

    else:
        aperture_df = pd.DataFrame({
            #'s': merged_data['aperture']['s'],
            #'name': merged_data['aperture']['name'],
            #'length': merged_data['aperture']['length'],
            'n': merged_data['aperture']['n'],
            'type': 'aperture'  # Label to distinguish between types
        })


    combined_df = pd.concat([collimator_df, aperture_df], ignore_index=True)

    # Convert the merged_data dictionary into a final DataFrame
    return collimator_df, aperture_df, from_part

def save_lossmap_to_json(merged_data, output_file):
    """
    Saves the merged lossmap data into a JSON file.
    """
    try:
        with open(output_file, 'w') as json_file:
            json.dump(merged_data, json_file, indent=4)
        print(f"Lossmap data successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving lossmap data to JSON file: {e}")

def prepare_lossmap_values(base_dir, output_dir, s_min, s_max, single_file, norm='none', total_sim_energy = 0):
    '''
    Function to process the losses data to produce datframe of collimator losses and aperture losses. It handles both json file from original script in htcondor and also the one produces from particles.hdf 
    '''
    lossmap_norms = ['none', 'max', 'coll_max','total', 'coll_total', 'tot_energy']
    if norm not in lossmap_norms:
        raise ValueError('norm must be in [{}]'.format(', '.join(lossmap_norms)))

    bin_w = 0.10
    nbins = int(np.ceil((s_max - s_min)/bin_w))
    
    warm_regions = FCC_EE_WARM_REGIONS_V25 #FCC_EE_WARM_REGIONS_V24


    if single_file:
        coll_group, ap_group, from_part = load_multiple_lossmaps(base_dir, output_dir, single_file=single_file)
    else: 
        coll_group, ap_group = load_multiple_lossmaps(base_dir, output_dir)

    if not from_part:

        ap_group = ap_group.groupby('name').agg(
            n=('n', 'sum'),           # Sum of 'n'
            s=('s', 'mean'),        # Average of 's'
        ).reset_index()
                
        ap_s = ap_group['s']
        aper_loss = ap_group['n']
        aper_edges = np.linspace(s_min, s_max, nbins + 1)

        coll_group = coll_group.groupby('name').agg(
            n=('n', 'sum'),           # Sum of 'n'
            s=('s', 'mean'),        # Average of 's'
            length=('length', 'mean')   # Sum of 'n' for each group
        ).reset_index()

    else:
        aper_edges = np.linspace(s_min, s_max, nbins)
        aper_loss = ap_group['n'].reindex(range(0, nbins-1), fill_value=0)
    
    coll_name = coll_group['name']
    coll_loss = coll_group['n']
    coll_s = coll_group['s']
    coll_length = coll_group['length']

    coll_end = coll_s + (coll_length/2)
    coll_start = coll_s - (coll_length/2)    
    
    if norm == 'total':
        norm_val = sum(coll_loss) + sum(aper_loss) # This is not exactly the energy of the initial beam due to the secondary particles/interaction where energy is lost
    elif norm == 'max':
        norm_val = max(max(coll_loss), max(aper_loss))
    elif norm == 'coll_total':
        norm_val = sum(coll_loss)
    elif norm == 'coll_max':
        norm_val = max(coll_loss)
    elif norm == 'none':
        norm_val = 1
    elif norm == 'tot_energy':
        total_beam_energy =  TOT_INJ_ENERGY #NB * KB * BEAM_ENERGY #REAL BEAM ENERGY AT Z
        total_sim_lost_energy = sum(coll_loss) + sum(aper_loss)
        lost_energy_fraction = total_sim_lost_energy / total_sim_energy
        total_lost_energy_joule = total_beam_energy * lost_energy_fraction #* EV_TO_JOULE
        norm_val = total_sim_lost_energy / total_lost_energy_joule

    if aper_loss.sum() > 0:
        
        aper_loss /= (norm_val * bin_w)

        if from_part:
            mask_warm = np.array([check_warm_loss(s, warm_regions)
                         for s in aper_edges[:-1]])

            ap_warm = aper_loss * mask_warm
            ap_cold = aper_loss * ~mask_warm

            ap_warm_pow = ap_warm * bin_w #/ norm_val
            ap_cold_pow = ap_cold * bin_w #/ norm_val

        else:
            mask_warm = np.array([check_warm_loss(s, warm_regions)
                                for s in ap_s])

            warm_loss = aper_loss * mask_warm
            cold_loss = aper_loss * ~mask_warm

            ap_warm = np.zeros(len(aper_edges) - 1)
            ap_cold = np.zeros(len(aper_edges) - 1)

            ap_warm_pow = np.zeros(len(aper_edges) - 1)
            ap_cold_pow = np.zeros(len(aper_edges) - 1)

            ap_indices = np.digitize(ap_s, aper_edges)

            np.add.at(ap_warm, ap_indices[mask_warm] - 1, warm_loss[mask_warm])
            np.add.at(ap_cold, ap_indices[~mask_warm] - 1, cold_loss[~mask_warm])

            
            warm_loss_pow = warm_loss * bin_w #/ norm_val
            cold_loss_pow = cold_loss * bin_w #/ norm_val

            np.add.at(ap_warm_pow, ap_indices[mask_warm] - 1, warm_loss_pow[mask_warm])
            np.add.at(ap_cold_pow, ap_indices[~mask_warm] - 1, cold_loss_pow[~mask_warm])
            
    else:
        aper_edges = [0]
        ap_warm = [0]
        ap_cold = [0]
        ap_warm_pow = [0]
        ap_cold_pow = [0]

    if coll_loss.sum() > 0:
        coll_pow = coll_loss / norm_val # fraction of the injected beam lost in Joule calculated as energy_lost_coll(eV)*(tot_energy_true_beam)(Joule)/(tot_energy_sim_beam(eV))
        coll_loss /= (norm_val * coll_length)
        zeros = np.full_like(coll_group.index, 0)  # Zeros to pad the bars
        coll_edges = np.dstack([coll_start, coll_start, coll_end, coll_end]).flatten()
        coll_loss = np.dstack([zeros, coll_loss, coll_loss, zeros]).flatten()
        coll_pow = np.dstack([zeros, coll_pow, coll_pow, zeros]).flatten()

    else:
        coll_edges = [0]
        coll_loss = [0]
        coll_pow = [0]    

    return coll_edges, coll_loss, coll_pow, aper_edges, ap_warm, ap_cold, ap_warm_pow, ap_cold_pow, total_sim_lost_energy

def plot_lossmaps(base_dir, output_dir, single_file, output_file, norm='none', tot_energy_full = 0):
    '''
    Function to plot lossmap in cleaning efficency, energy lost and zoom in IPG and IPF.
    '''
    # Load twiss parameters
    #twiss = pd.read_json(twiss, orient='split')
    #s_min, s_max = twiss.s.min(), twiss.s.max()
    s_min, s_max = 0, 90658.50572430571

    coll_edges, coll_loss, coll_pow, aper_edges, ap_warm, ap_cold, ap_warm_pow, ap_cold_pow, tot_energy = prepare_lossmap_values(base_dir, output_dir, s_min, s_max, single_file, norm, tot_energy_full)
    
    fig, ax = plt.subplots(figsize=(18, 6))

    # UNCOMMENT FOR LLSS COMMON LATTICE

    x_shift = 22664.626431 #2032.000 + 1400.000 + 9616.175 
    x_wrap = s_max
    
    if coll_edges is None or len(coll_edges) == 0:  # Check if aper_edges is empty
        print("Warning: aper_edges is empty. Skipping operation.")
        return  # or handle the empty case appropriately
    else:
        coll_edges = [(edge + x_shift) % x_wrap for edge in coll_edges]
    
    if aper_edges is None or len(aper_edges) == 0:   # Check if aper_edges is empty
        print("Warning: aper_edges is empty. Skipping operation.")
        return  # or handle the empty case appropriately
    else:
        #aper_edges = (aper_edges + x_shift)  % x_wrap
        aper_edges = [(edge + x_shift) % x_wrap for edge in aper_edges]

    lw=1
    if np.sum(coll_loss) == 0:  # Check if any coll_loss value is greater than zero
        print("coll_loss is zero; skipping plot.")
    else:
        ax.fill_between(coll_edges, coll_loss, step='pre', color='k', zorder=9)
        ax.step(coll_edges, coll_loss, color='k', lw=lw, zorder=10, label='Collimator losses')
    
    if np.sum(ap_warm) == 0 and np.sum(ap_cold) == 0:
        print("No losses in the aperture; skipping plot.")
    else:
        # Plot warm losses
        ax.fill_between(aper_edges[:-1], ap_warm, step='post', color='r', zorder=9)
        ax.step(aper_edges[:-1], ap_warm, where='post', color='red', label='Warm', linewidth=1)

        # Plot cold losses
        ax.fill_between(aper_edges[:-1], ap_cold, step='post', color='b', zorder=9)
        ax.step(aper_edges[:-1], ap_cold, where='post', color='blue', label='Cold', linewidth=1)

    plot_margin = 500
    ax.set_xlim(s_min - plot_margin, s_max + plot_margin)

    ax.yaxis.grid(visible=True, which='major', zorder=0)
    ax.yaxis.grid(visible=True, which='minor', zorder=0)

    ax.set_xlabel('s [m]')
    ax.set_yscale('log', nonpositive='clip')
    ax.set_ylabel('Cleaning inefficiency[$m^{-1}$]' )
    #ax.set_title('Lossmap with Exciter')
    ax.legend(loc='upper right')  # Corrected 'upper rigth' to 'upper right' 
    ax.grid()

    plt.tick_params(axis='both', which='major', labelsize=14) 
    # Finalize and save the plot
    plt.savefig(os.path.join(output_dir,f'{output_file}_lossmap.png'),bbox_inches='tight') 
    plt.close()
    # POWER PLOT
    fig, ax_pow = plt.subplots(figsize=(18, 6))
    #fig, ax_pow = plt.subplots(figsize=(5, 7))

    lw=1
    if np.sum(coll_pow) == 0:  # Check if any coll_loss value is greater than zero
        print("coll_loss is zero; skipping plot.")
    else:
        ax_pow.fill_between(coll_edges, coll_pow, step='pre', color='k', zorder=9)
        ax_pow.step(coll_edges, coll_pow, color='k', lw=lw, zorder=10, label='Coll')

    if np.sum(ap_warm_pow) == 0 and np.sum(ap_cold_pow) == 0:
        print("No losses in the aperture; skipping plot.")
    else:
        # Plot warm losses
        ax_pow.fill_between(aper_edges[:-1], ap_warm_pow, step='post', color='r', zorder=9)
        ax_pow.step(aper_edges[:-1], ap_warm_pow, where='post', color='red', label='Warm', linewidth=1)

        # Plot cold losses
        ax_pow.fill_between(aper_edges[:-1], ap_cold_pow, step='post', color='b', zorder=9)
        ax_pow.step(aper_edges[:-1], ap_cold_pow, where='post', color='blue', label='Cold', linewidth=1)

    plot_margin = 500
    ax_pow.set_xlim(s_min - plot_margin, s_max + plot_margin)
    ax_pow.set_ylim(1,2e4)

    ax_pow.yaxis.grid(visible=True, which='major', zorder=0)
    ax_pow.yaxis.grid(visible=True, which='minor', zorder=0)

    ax_pow.set_xlabel('s [m]', fontsize=16)
    ax_pow.set_yscale('log', nonpositive='clip')
    ax_pow.set_ylabel('Energy Lost [$J$]', fontsize=16)
    #ax_pow.set_title('Lossmap in power with Exciter')
    ax_pow.legend(loc='upper right', fontsize=12)  # Corrected 'upper rigth' to 'upper right'
    ax_pow.grid()
    plt.tick_params(axis='both', which='major', labelsize=16) 
    # Finalize and save the plot
    plt.savefig(os.path.join(output_dir,f'{output_file}_pow.png'),bbox_inches='tight') 
    plt.close()
    # Create a zoomed-in plot
    fig_zoom_IPG, ax_zoom_IPG = plt.subplots(figsize=(10, 5))

    # Define the zoom region for the x and y axes
    x_min, x_max = 44500, 46500 #  around IPG  
    #x_min, x_max = 67300, 68700 #  around IPJ  
    #y_min, y_max = 0, 500

    if np.sum(coll_pow) == 0:  # Check if any coll_loss value is greater than zero
        print("coll_loss is zero; skipping plot.")
    else:
        ax_zoom_IPG.fill_between(coll_edges, coll_pow, step='pre', color='k', zorder=9)
        ax_zoom_IPG.step(coll_edges, coll_pow, color='k', lw=lw, zorder=10, label='Collimator losses')
    
    if np.sum(ap_warm_pow) == 0 and np.sum(ap_cold_pow) == 0:
        print("No losses in the aperture; skipping plot.")
    else:
        # Plot warm losses
        ax_zoom_IPG.fill_between(aper_edges[:-1], ap_warm_pow, step='post', color='r', zorder=9)
        ax_zoom_IPG.step(aper_edges[:-1], ap_warm_pow, where='post', color='red', label='Warm losses', linewidth=1)

        # Plot cold losses
        ax_zoom_IPG.fill_between(aper_edges[:-1], ap_cold_pow, step='post', color='b', zorder=9)
        ax_zoom_IPG.step(aper_edges[:-1], ap_cold_pow, where='post', color='blue', label='Cold losses', linewidth=1)
    
    plot_margin = 500
    ax_zoom_IPG.set_xlim(s_min - plot_margin, s_max + plot_margin)

    ax_zoom_IPG.yaxis.grid(visible=True, which='major', zorder=0)
    ax_zoom_IPG.yaxis.grid(visible=True, which='minor', zorder=0)

    # Set the limits for zooming in
    ax_zoom_IPG.set_xlim(x_min, x_max)
    ax_zoom_IPG.set_ylim(0.01, 1e10)

    # Add labels, title, and legend for the zoomed plot
    ax_zoom_IPG.set_xlabel('s[m]')
    ax_zoom_IPG.set_yscale('log', nonpositive='clip')
    #ax_zoom_IPG.set_ylabel('Cleaning inefficiency[$m^{-1}$]')
    ax_zoom_IPG.set_ylabel('Energy Lost [$J$]')
    ax_zoom_IPG.set_title('Zoom at IPG: Lossmap with exciter')
    ax_zoom_IPG.legend(loc='upper right')
    ax_zoom_IPG.grid()

    # Finalize and save the zoomed plot
    plt.savefig(os.path.join(output_dir,f'{output_file}_zoom_IPG.png'))
    plt.close()
    # Zoom in other range
    fig_zoom_IPF, ax_zoom_IPF = plt.subplots(figsize=(10, 5))

    # Define the zoom region for the x and y axes
    x_min, x_max = 33000, 35500  # around collimation insertion IPF
    #y_min, y_max = 0, 500

    if np.sum(coll_pow) == 0:  # Check if any coll_loss value is greater than zero
        print("coll_loss is zero; skipping plot.")
    else:
        ax_zoom_IPF.fill_between(coll_edges, coll_pow, step='pre', color='k', zorder=9)
        ax_zoom_IPF.step(coll_edges, coll_pow, color='k', lw=lw, zorder=10, label='Collimator losses')
    
    if np.sum(ap_warm_pow) == 0 and np.sum(ap_cold_pow) == 0:
        print("No losses in the aperture; skipping plot.")
    else:
        # Plot warm losses
        ax_zoom_IPF.fill_between(aper_edges[:-1], ap_warm_pow, step='post', color='r', zorder=9)
        ax_zoom_IPF.step(aper_edges[:-1], ap_warm_pow, where='post', color='red', label='Warm losses', linewidth=1)

        # Plot cold losses
        ax_zoom_IPF.fill_between(aper_edges[:-1], ap_cold_pow, step='post', color='b', zorder=9)
        ax_zoom_IPF.step(aper_edges[:-1], ap_cold_pow, where='post', color='blue', label='Cold losses', linewidth=1)
    
    plot_margin = 500
    ax_zoom_IPF.set_xlim(s_min - plot_margin, s_max + plot_margin)

    ax_zoom_IPF.yaxis.grid(visible=True, which='major', zorder=0)
    ax_zoom_IPF.yaxis.grid(visible=True, which='minor', zorder=0)

    # Set the limits for zooming in
    ax_zoom_IPF.set_xlim(x_min, x_max)
    ax_zoom_IPF.set_ylim(0.01, 1e10)

    # Add labels, title, and legend for the zoomed plot
    ax_zoom_IPF.set_xlabel('s[m]')
    ax_zoom_IPF.set_yscale('log', nonpositive='clip')
    #ax_zoom_IPF.set_ylabel('Cleaning inefficiency[$m^{-1}$]')
    ax_zoom_IPF.set_ylabel('Energy Lost [$J$]')
    ax_zoom_IPF.set_title('Zoom at IPF: Lossmap with exciter')
    ax_zoom_IPF.legend(loc='upper right')
    ax_zoom_IPF.grid()

    # Finalize and save the zoomed plot
    plt.savefig(os.path.join(output_dir,f'{output_file}_zoom_IPF.png'))
    plt.close()

    return tot_energy

def collimators_names(coll_dat):
        
    data = {
        "name": [],
        "opening": [],
        "material": [],  # New column for "mat."
        "length": [],
        "angle": [],
        "offset": []
    }

    # Parse the file
    with open(coll_dat, 'r') as file:
        start_parsing = False
        for line in file:
            # Strip whitespace from line
            line = line.strip()
            
            # Start parsing when we reach the relevant section
            if "name" in line:
                start_parsing = True
                continue
            
            # Stop parsing at the end of the collimators section
            if "SETTINGS" in line:
                break
            
            # Parse only relevant lines that contain collimator data
            if start_parsing and line:
                parts = line.split()
                
                # Ensure there are enough parts to avoid index errors
                if len(parts) >= 6:
                    # Append the relevant data
                    data["name"].append(parts[0])
                    data["opening"].append(parts[1])
                    data["material"].append(parts[2])       # New column for material
                    data["length"].append(float(parts[3]))  # Convert to float
                    data["angle"].append(float(parts[4]))   # Convert to float
                    data["offset"].append(float(parts[5]))  # Convert to float

    # Create DataFrame
    df = pd.DataFrame(data)

    return df['name'].values

def collimators_names_json(coll_dat_json):
    # Load the JSON data
    with open(coll_dat_json, 'r') as file:
        data = json.load(file)
    
    # Extract the families and collimators sections
    families = data["families"]
    collimators = data["Collimators"]
    
    # Initialize data structure for DataFrame
    df_data = {
        "name": [],
        "gap": [],
        "stage": [],
        "material": [],
        "length": [],
        "angle": []
    }
    
    # Iterate over collimators and map their attributes from the families section
    for coll_name, coll_info in collimators.items():
        family_name = coll_info["family"]
        
        # Ensure the family exists in the families section
        if family_name in families:
            family_attrs = families[family_name]
            
            # Append the collimator data to the DataFrame structure
            df_data["name"].append(coll_name)
            df_data["gap"].append(family_attrs["gap"])
            df_data["stage"].append(family_attrs["stage"])
            df_data["material"].append(family_attrs["material"])
            df_data["length"].append(float(family_attrs["length"]))  # Convert to float
            df_data["angle"].append(float(family_attrs["angle"]))    # Convert to float
    
    # Create the DataFrame
    df = pd.DataFrame(df_data)
    
    # Return the 'name' column values as a NumPy array
    return df["name"].values

def main(base_dir):

    output_dir = base_dir
    
    part_files = ["part_merged.hdf", "part.hdf"]
    part_file = None
    for file in part_files:
        if os.path.exists(os.path.join(base_dir, file)):
            part_file = file
            break

    if part_file:
        df_part = pd.read_hdf(os.path.join(base_dir, part_file), key="particles")
        print(f"Loaded {part_file}")
    else:
        raise FileNotFoundError("Neither part_merged.hdf nor part.hdf was found in the output directory.")
    
    total_sim_energy = BEAM_ENERGY*len(df_part[df_part['parent_particle_id'] == df_part['particle_id']])

    single_file = 'merged_lossmap_full.json'
    output_file = 'merged_lossmap_full'
    tot_energy_lost = plot_lossmaps(output_dir, output_dir, single_file, output_file, norm='tot_energy', tot_energy_full=total_sim_energy)

    with open(os.path.join(output_dir,'loss.txt'), "w") as file:
        file.write(f"Total Energy: {total_sim_energy}\n")
        file.write(f"Loss total: {tot_energy_lost}\n")

        '''for i in turns:
            single_file = f'merged_lossmap_turn_{i}.json'
            output_file = f'merged_lossmap_turn_{i}'
            tot_energy_turn = plot_lossmaps(output_dir, output_dir, twiss, single_file, output_file, norm='coll_turn',tot_energy_full=tot_energy_full)
            file.write(f"Loss at turn {i}: {tot_energy_turn}\n")'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Track particles with a kicker and plot the results.')
    parser.add_argument('--base_dir', type=str, required=True, help='Path to lossmap files, for ex.: dataset/new_vertical/3_turns/ver_phase_90_3turns.')
    args = parser.parse_args()
    # Call the main function with parsed arguments
    main(args.base_dir)