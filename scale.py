import os
import time
import configparser as cp
import run_nets as r
from absl import flags
from absl import app

FLAGS = flags.FLAGS
#name of flag | default | explanation
flags.DEFINE_string("arch_config","./configs/YoloTiny_Fixed_Sram.cfg","file where we are getting our architechture from")

flags.DEFINE_string("network","./topologies/conv_nets/yolo_v3_tiny.csv","topology that we are reading")


class scale:
    def __init__(self, sweep = False, save = False):
        self.sweep = sweep
        self.save_space = save

    def parse_config(self):
        general = 'general'
        arch_sec = 'architecture_presets'
        net_sec  = 'network_presets'
       # config_filename = "./scale.cfg"
        config_filename = FLAGS.arch_config
        print("Using Architechture from ",config_filename)

        config = cp.ConfigParser()
        config.read(config_filename)

        ## Read the run name
        self.run_name = config.get(general, 'run_name')

        ## Read the architecture_presets
        ## Array height min, max
        ar_h = config.get(arch_sec, 'ArrayHeight').split(',')
        self.ar_h_min = ar_h[0].strip()
        print(self.ar_h_min)

        if len(ar_h) > 1:
            self.ar_h_max = ar_h[1].strip()
        #print("Min: " + ar_h_min + " Max: " + ar_h_max)

        ## Array width min, max
        ar_w = config.get(arch_sec, 'ArrayWidth').split(',')
        self.ar_w_min = ar_w[0].strip()

        if len(ar_w) > 1:
            self.ar_w_max = ar_w[1].strip()

        ## IFMAP SRAM buffer min, max
        ifmap_sram = config.get(arch_sec, 'IfmapSramSz').split(',')
        self.isram_min = ifmap_sram[0].strip()

        if len(ifmap_sram) > 1:
            self.isram_max = ifmap_sram[1].strip()


        ## FILTER SRAM buffer min, max
        filter_sram = config.get(arch_sec, 'FilterSramSz').split(',')
        self.fsram_min = filter_sram[0].strip() 

        if len(filter_sram) > 1:
            self.fsram_max = filter_sram[1].strip()


        ## OFMAP SRAM buffer min, max
        ofmap_sram = config.get(arch_sec, 'OfmapSramSz').split(',')
        self.osram_min = ofmap_sram[0].strip()


        if len(ofmap_sram) > 1:
            self.osram_max = ofmap_sram[1].strip()

        ## Total_sram_size
        ofmap_sram = config.get(arch_sec, 'TotalSramSz').split(',')
        self.Total_sram_size = ofmap_sram[0].strip()

        self.dataflow= config.get(arch_sec, 'Dataflow')

        ifmap_offset = config.get(arch_sec, 'IfmapOffset')
        self.ifmap_offset = int(ifmap_offset.strip())

        filter_offset = config.get(arch_sec, 'FilterOffset')
        self.filter_offset = int(filter_offset.strip())

        ofmap_offset = config.get(arch_sec, 'OfmapOffset')
        self.ofmap_offset = int(ofmap_offset.strip())

        word_size_bytes = config.get(arch_sec, 'WordSizeByte')
        self.word_size_bytes = int(word_size_bytes.strip())

        ## Read network_presets
        ## For now that is just the topology csv filename
        #topology_file = config.get(net_sec, 'TopologyCsvLoc')
        #self.topology_file = topology_file.split('"')[1]     #Config reads the quotes as wells
        self.topology_file= FLAGS.network

    def run_scale(self):
        self.parse_config()

        if self.sweep == False:
            self.run_once()
        else:
            self.run_sweep()


    def run_once(self):

        df_string = "Output Stationary"
        if self.dataflow == 'ws':
            df_string = "Weight Stationary"
        elif self.dataflow == 'is':
            df_string = "Input Stationary"

        print("====================================================")
        print("******************* DSE START **********************")
        print("====================================================")
        print("Array Size: \t" + str(self.ar_h_min) + "x" + str(self.ar_w_min))
        print("SRAM IFMAP: \t" + str(self.isram_min))
        print("SRAM Filter: \t" + str(self.fsram_min))
        print("SRAM OFMAP: \t" + str(self.osram_min))
        print("CSV file path: \t" + self.topology_file)
        print("Dataflow: \t" + df_string)
        print("====================================================")

        net_name = self.topology_file.split('/')[-1].split('.')[0]
        #print("Net name = " + net_name)
        offset_list = [self.ifmap_offset, self.filter_offset, self.ofmap_offset]

        r.run_net(  ifmap_sram_size  = int(self.isram_min),
                    filter_sram_size = int(self.fsram_min),
                    ofmap_sram_size  = int(self.osram_min),
                    array_h = int(self.ar_h_min),
                    array_w = int(self.ar_w_min),
                    net_name = net_name,
                    data_flow = self.dataflow,
                    topology_file = self.topology_file,
                    offset_list=offset_list,
                    word_size_bytes = self.word_size_bytes
                )
        self.cleanup()
        print("************ SCALE SIM Run Complete ****************")


    def cleanup(self):
        if not os.path.exists("./outputs/"):
            os.system("mkdir ./outputs")

        net_name = self.topology_file.split('/')[-1].split('.')[0]

        path = "./output/scale_out"
        if self.run_name == "":
            path = "./outputs/" + net_name +"_"+ self.dataflow
        else:
            path = "./outputs/" + self.run_name

        if not os.path.exists(path):
            os.system("mkdir " + path)
        else:
            t = time.time()
            new_path= path + "_" + str(t)
            os.system("mv " + path + " " + new_path)
            os.system("mkdir " + path)

        cmd = "mv *.csv " + path
        os.system(cmd)

        cmd = "mkdir " + path +"/layer_wise"
        os.system(cmd)

        cmd = "mv " + path +"/*sram* " + path +"/layer_wise"
        os.system(cmd)

        cmd = "mv " + path +"/*dram* " + path +"/layer_wise"
        os.system(cmd)

        if self.save_space == True:
            cmd = "rm -rf " + path +"/layer_wise"
            os.system(cmd)


    def run_sweep(self):

        # all_data_flow_list = ['ws','is']
        all_data_flow_list = ['os']
        # all_arr_dim_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        # all_sram_sz_list = [256, 512, 1024]

        # all_ifmap_sram_sz_list = [1, 2, 4, 8, 16, 32, 64]
        # all_ofmap_sram_sz_list = [1, 2, 4, 8, 16, 32, 64]
        # all_filt_sram_sz_list = [1, 2, 4, 8, 16, 32, 64]

        data_flow_list = all_data_flow_list

        #arr_w_list = list(reversed(arr_h_list))

        net_name = self.topology_file.split('/')[-1].split('.')[0]
        total_sram_sz = int(self.Total_sram_size)

        isram_begin , fsram_begin, osram_begin = int(self.isram_min), int(self.fsram_min), int(self.osram_min)
        for df in data_flow_list:
            self.dataflow = df
            # for i in range(len(arr_h_list)):
            # self.ar_h_min = arr_h_list
            # self.ar_w_min = arr_w_list

            # self.run_name = net_name + "_" + df + "_" + str(self.ar_h_min) + "x" + str(self.ar_w_min)
            # print(str(self.isram_min, self.fsram_min, self.isram_min)+"\n")
            # print(str(self.isram_max, self.fsram_max, self.isram_max)+"\n")
            # isram_end   , fsram_end  ,osram_end   =   
            if(df == 'is'):
                isram_sz = isram_begin
                osram_sz = osram_begin
                while (isram_sz < int(self.isram_max)):
                    fsram_sz = total_sram_sz - isram_sz
                    self.fsram_min = str(fsram_sz)
                    self.run_name = net_name + "_" + df + "_" + str(isram_sz) + "x" + str(fsram_sz) + "x" + str(osram_sz)
                    print(self.run_name)
                    self.run_once()
                    self.osram_min = str(osram_sz)
                    isram_sz *= 2
                    self.isram_min = str(isram_sz)

                fsram_sz = fsram_begin
                osram_sz = osram_begin
                while (fsram_sz < int(self.isram_max)):
                    isram_sz = total_sram_sz - fsram_sz
                    self.isram_min = str(isram_sz)
                    if (isram_sz == fsram_sz): 
                        break
                    self.run_name = net_name + "_" + df + "_" + str(isram_sz) + "x" + str(fsram_sz) + "x" + str(osram_sz)
                    print(self.run_name)
                    self.run_once()
                    self.osram_min = str(osram_sz)
                    # print(osram_sz)
                    fsram_sz *= 2
                    self.fsram_min = str(fsram_sz)
                 
            elif(df == 'ws'):

                isram_sz = isram_begin
                osram_sz = osram_begin
                while (isram_sz < int(self.isram_max)):
                    fsram_sz = total_sram_sz - isram_sz
                    self.fsram_min = str(fsram_sz)
                    self.run_name = net_name + "_" + df + "_" + str(isram_sz) + "x" + str(fsram_sz) + "x" + str(osram_sz)
                    print(self.run_name)
                    self.run_once()
                    self.osram_min = str(osram_sz)
                    isram_sz *= 2
                    self.isram_min = str(isram_sz)

                fsram_sz = fsram_begin
                osram_sz = osram_begin
                while (fsram_sz < int(self.isram_max)):
                    isram_sz = total_sram_sz - fsram_sz
                    self.isram_min = str(isram_sz)
                    if (isram_sz == fsram_sz): 
                        break
                    self.run_name = net_name + "_" + df + "_" + str(isram_sz) + "x" + str(fsram_sz) + "x" + str(osram_sz)
                    print(self.run_name)
                    self.run_once()
                    self.osram_min = str(osram_sz)
                    # print(osram_sz)
                    fsram_sz *= 2
                    self.fsram_min = str(fsram_sz) 
                

            elif(df == 'os'):
                osram_sz = osram_begin
                while (osram_sz <= int(self.osram_max)):
                    fsram_sz = fsram_begin
                    while (fsram_sz <= int(self.fsram_max)):
                        isram_sz = isram_begin
                        while (isram_sz <= int(self.isram_max)):
                            if(2*osram_sz+2*isram_sz+2*fsram_sz < total_sram_sz):
                                break
                            self.run_name = net_name + "_" + df + "_" + str(isram_sz) + "x" + str(fsram_sz) + "x" + str(osram_sz)
                            print(self.run_name)
                            self.run_once()
                            # print(osram_sz)
                            isram_sz *= 2
                            self.isram_min = str(isram_sz)
                        self.fsram_min = str(fsram_sz)
                        fsram_sz *= 2
                osram_sz *= 2
                self.osram_min = str(osram_sz)


                # while (isram_sz <= int(self.isram_max)):
                #     while (fsram_sz <= int(self.fsram_max)):
                #         osram_sz = osram_begin
                #         while (osram_sz <= int(self.osram_max)):
                #             if(df == 'is' and (2*isram_sz + fsram_sz >= total_sram_sz)):
                #                 break
                #             if(df == 'ws' and (isram_sz + 2*fsram_sz >= total_sram_sz)):
                #                 break
                #             if(df == 'os' and (isram_sz + fsram_sz + 2*osram_sz >= total_sram_sz)):
                #                 break 
                #             self.run_name = net_name + "_" + df + "_" + str(isram_sz) + "x" + str(fsram_sz) + "x" + str(osram_sz)
                #             print(self.run_name)
                #             self.run_once()
                #             osram_sz *= 2
                #             self.osram_min = str(osram_sz)
                #             # print(osram_sz)
                #         fsram_sz *= 2
                #         self.fsram_min = str(fsram_sz)
                #     isram_sz *= 2
                #     self.isram_min = str(isram_sz)


def main(argv):
    s = scale(save = False, sweep = True)
    s.run_scale()

if __name__ == '__main__':
  app.run(main)
'''
if __name__ == "__main__":
    s = scale(save = False, sweep = False)
    s.run_scale()
'''
