import os.path
import math
from tqdm import tqdm


def prune(input_list):
    l = []

    for e in input_list:
        e = e.strip()
        if e != '' and e != ' ':
            l.append(e)

    return l


def dram_trace_read_v2(
        sram_sz         = 512 * 1024,       #这里的大小应该就是Byte
        word_sz_bytes   = 1,                #一个字几个字节
        min_addr = 0, max_addr=1000000,
        default_read_bw = 8,               # this is arbitrary
        sram_trace_file = "sram_log.csv",
        dram_trace_file = "dram_log.csv"
    ):

    total_dram_acc_times = 0        # Dram访存次数
    t_fill_start    = -1
    t_drain_start = 0           # 开始计算的时间，从计算时间开始倒推填充的起始时间
    init_bw = default_read_bw   # Taking an arbitrary initial bw of 4 bytes per cycle
                                # SRAM的带宽，一个cycle能取多少个word的数，在我们的cgra中是8
    

    sram = set()

    # 根据sram中的数倒推load的时间
    sram_requests = open(sram_trace_file, 'r')
    dram          = open(dram_trace_file, 'w')

    #for entry in tqdm(sram_requests):
    # 每次遍历一行（也就是一个cycle）的sram数据，放在一个对应的list中
    for entry in sram_requests:
        elems = entry.strip().split(',')
        elems = prune(elems)
        elems = [float(x) for x in elems]

        clk = elems[0]  #第一个元素是时钟信息

        # 逐一添加元素
        for e in range(1, len(elems)):
            
            # 通过此地址判断是否是有效，并且不再sram中则需要去Dram中取
            if (elems[e] not in sram) and (elems[e] >= min_addr) and (elems[e] < max_addr):

                # Used up all the unique data in the SRAM?
                # 根据阵列到SRAM的请求，取不大于sram容量，最多的数。因为计算是理想的每个周期都在算，
                # 所以尽量按照sram的大小进行分块
                #len(sram)是元素数量
                if (len(sram) + 1)*word_sz_bytes > sram_sz:

                    # 开始计算填充满需要的时间
                    if t_fill_start == -1:
                        # t_fill_start = t_drain_start - math.ceil(len(sram) / (init_bw * word_sz_bytes))
                        t_fill_start = t_drain_start - math.ceil(len(sram) / (init_bw))

                    # Generate the filling trace from time t_fill_start to t_drain_start
                    # 需要填充的时间
                    cycles_needed   = t_drain_start - t_fill_start
                    # words_per_cycle = math.ceil(len(sram) / (cycles_needed * word_sz_bytes))
                    words_per_cycle = math.ceil(len(sram) / cycles_needed)

                    c = t_fill_start

                    while len(sram) > 0:
                        trace = str(c) + ", "

                        for _ in range(words_per_cycle):
                            if len(sram) > 0:
                                p = sram.pop()
                                trace += str(p) + ", "
                                total_dram_acc_times += 1

                        trace += "\n"
                        dram.write(trace)
                        c += 1

                    t_fill_start    = t_drain_start
                    t_drain_start   = clk

                # Add the new element to sram
                sram.add(elems[e])


    # 这一部分与上述代码的作用相同，只是为了解决sram中剩余的元素
    if len(sram) > 0:
        if t_fill_start == -1:
            # t_fill_start = t_drain_start - math.ceil(len(sram) / (init_bw * word_sz_bytes))
            t_fill_start = t_drain_start - math.ceil(len(sram) / init_bw )

        # Generate the filling trace from time t_fill_start to t_drain_start
        cycles_needed = t_drain_start - t_fill_start
        # words_per_cycle = math.ceil(len(sram) / (cycles_needed * word_sz_bytes))
        words_per_cycle = math.ceil(len(sram) / cycles_needed)

        c = t_fill_start

        while len(sram) > 0:
            trace = str(c) + ", "

            for _ in range(words_per_cycle):
                if len(sram) > 0:
                    p = sram.pop()
                    trace += str(p) + ", "
                    total_dram_acc_times += 1

            trace += "\n"
            dram.write(trace)
            c += 1

    sram_requests.close()
    dram.close()
    return total_dram_acc_times 

def dram_trace_write(ofmap_sram_size = 64,
                     data_width_bytes = 1,
                     default_write_bw = 8,                     # this is arbitrary
                     sram_write_trace_file = "sram_write.csv",
                     dram_write_trace_file = "dram_write.csv"):

    traffic = open(sram_write_trace_file, 'r')
    trace_file  = open(dram_write_trace_file, 'w')

    last_clk = 0
    clk = 0
    total_dram_acc_times = 0

    sram_buffer = [set(), set()]
    filling_buf     = 0
    draining_buf    = 1

    for row in traffic:
        elems = row.strip().split(',')
        elems = prune(elems)
        elems = [float(x) for x in elems]

        clk = elems[0]

        # If enough space is in the filling buffer
        # Keep filling the buffer
        if (len(sram_buffer[filling_buf]) + (len(elems) - 1) * data_width_bytes ) < ofmap_sram_size:
            for i in range(1,len(elems)):
                sram_buffer[filling_buf].add(elems[i])

        # Filling buffer is full, spill the data to the other buffer
        else:
            # If there is data in the draining buffer
            # drain it
            #print("Draining data. CLK = " + str(clk))
            if len(sram_buffer[draining_buf]) > 0:
                delta_clks = clk - last_clk
                # data_per_clk = math.ceil(len(sram_buffer[draining_buf]) / delta_clks)
                data_per_clk = default_write_bw
                #print("Data per clk = " + str(data_per_clk))

                # Drain the data
                c = last_clk + 1
                while len(sram_buffer[draining_buf]) > 0:
                    trace = str(c) + ", "
                    c += 1
                    for _ in range(int(data_per_clk)):
                        if len(sram_buffer[draining_buf]) > 0:
                            addr = sram_buffer[draining_buf].pop()
                            trace += str(addr) + ", "
                            total_dram_acc_times += 1

                    trace_file.write(trace + "\n")
                last_clk = c

            #Swap the ids for drain buffer and fill buffer
            tmp             = draining_buf
            draining_buf    = filling_buf
            filling_buf     = tmp

            #Set the last clk value
            # last_clk = clk

            #Fill the new data now
            for i in range(1,len(elems)):
                sram_buffer[filling_buf].add(elems[i])

    #Drain the last fill buffer
    reasonable_clk = clk
    if len(sram_buffer[draining_buf]) > 0:
        #delta_clks = clk - last_clk
        #data_per_clk = math.ceil(len(sram_buffer[draining_buf]) / delta_clks)
        data_per_clk = default_write_bw
        #print("Data per clk = " + str(data_per_clk))

        # Drain the data
        c = last_clk + 1
        while len(sram_buffer[draining_buf]) > 0:
            trace = str(c) + ", "
            c += 1
            for _ in range(int(data_per_clk)):
                if len(sram_buffer[draining_buf]) > 0:
                    addr = sram_buffer[draining_buf].pop()
                    trace += str(addr) + ", "
                    total_dram_acc_times += 1


            trace_file.write(trace + "\n")
            reasonable_clk = max(c, clk)

    if len(sram_buffer[filling_buf]) > 0:
        data_per_clk = default_write_bw

        # Drain the data
        c = reasonable_clk + 1
        while len(sram_buffer[filling_buf]) > 0:
            trace = str(c)+ ", "
            c += 1
            for _ in range(int(data_per_clk)):
                if len(sram_buffer[filling_buf]) > 0:
                    addr = sram_buffer[filling_buf].pop()
                    trace += str(addr) + ", "
                    total_dram_acc_times += 1

            trace_file.write(trace + "\n")


    #All traces done
    traffic.close()
    trace_file.close()
    return total_dram_acc_times

if __name__ == "__main__":
    dram_trace_read_v2(min_addr=0, max_addr=1000000, dram_trace_file="ifmaps_dram_read.csv")
    dram_trace_read_v2(min_addr=1000000, max_addr=100000000, dram_trace_file="filter_dram_read.csv")
        #dram_trace_read(filter_sram_sz=1024, ifmap_sram_sz=1024, sram_trace_file="sram_read.csv")
        #dram_trace_write(ofmap_sram_size=1024,sram_write_trace_file="yolo_tiny_layer1_write.csv", dram_write_trace_file="yolo_tiny_layer1_dram_write.csv")
