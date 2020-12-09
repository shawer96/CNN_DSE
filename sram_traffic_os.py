import math 
from tqdm import tqdm

neg_inf = -1 * math.pow(2,32)

def sram_traffic(
        dimension_rows=4,
        dimension_cols=4,
        ifmap_h=7, ifmap_w=7,
        filt_h=3, filt_w=3,
        num_channels=3,
        strides=1, num_filt=8,
        ofmap_base=2000000, filt_base=1000000, ifmap_base=0,
        sram_read_trace_file="sram_read.csv",
        sram_write_trace_file="sram_write.csv"
):


    # Dimensions of output feature map channel
    E_h = (ifmap_h - filt_h + strides) / strides    #ofmap的高
    E_w = (ifmap_w - filt_w + strides) / strides    #ofmap的宽
    
    # Number of pixels in one convolution window
    px_per_conv_window = filt_h * filt_w * num_channels
    r2c = px_per_conv_window

    # Total number of ofmap px across all channels
    num_ofmap_px = E_h * E_w * num_filt
    e2  = E_h * E_w
    e2m = num_ofmap_px
    
    # Variables to calculate folds in runtime
    # 单个output channel在一列上展开, 最理想情况是d-rows = e2, 也就是每个pixels都有一个计算单元, 但实际上不可能
    num_h_fold = math.ceil(e2 / dimension_rows)
    # 沿着水平方向展开的还是oc
    num_v_fold = math.ceil(num_filt/dimension_cols)     #每一列需要计算几个oc

    cycles = 0

    read_cycles, util = gen_read_trace(
                            cycle = cycles,
                            dim_rows = dimension_rows,
                            dim_cols = dimension_cols,
                            num_v_fold = int(num_v_fold),
                            num_h_fold = int(num_h_fold),
                            ifmap_h = ifmap_h, ifmap_w= ifmap_w,
                            filt_h= filt_h, filt_w= filt_w,
                            num_channels= num_channels, stride=strides,
                            ofmap_h= int(E_h), ofmap_w= int(E_w), num_filters = num_filt,
                            filt_base= filt_base, ifmap_base= ifmap_base,
                            sram_read_trace_file= sram_read_trace_file
                            )

    write_cycles = gen_write_trace(
                        cycle = cycles,
                        dim_rows = dimension_rows,
                        dim_cols = dimension_cols,
                        #num_v_fold = int(num_v_fold),
                        #num_h_fold = int(num_h_fold),
                        ofmap_h = int(E_h), ofmap_w = int(E_w),
                        num_filters = num_filt,
                        ofmap_base = ofmap_base,
                        conv_window_size = r2c,
                        sram_write_trace_file = sram_write_trace_file
                        )

    cycles = max(read_cycles, write_cycles)
    str_cycles = str(cycles)
    return(str_cycles, util)
# End of sram_traffic()

        
def gen_read_trace(
        cycle = 0,
        dim_rows = 4, 
        dim_cols = 4,
        num_v_fold = 1,
        num_h_fold = 1,
        ifmap_h = 7, ifmap_w = 7,
        filt_h = 3, filt_w =3,
        num_channels = 3, stride = 1,
        ofmap_h =5, ofmap_w = 5, num_filters = 8, 
        filt_base = 1000000, ifmap_base = 0,
        sram_read_trace_file = "sram_read.csv",
        #sram_write_trace_file = "sram_write.csv"
):
    # Layer specific variables
    r2c = filt_h * filt_w * num_channels
    rc = filt_w * num_channels
    hc = ifmap_w * num_channels
    e2 = ofmap_h * ofmap_w
    #num_ofmap_px = e2 * num_filters
    
    # Tracking variables
    local_cycle     = 0
    #remaining_px    = e2           # Need tracking for individual v folds
    #remaining_px     = []
    remaining_filt  = num_filters
    ifmap_done      = False
    filt_done       = False
    row_base_addr   = []    # 第r行的ifmap第一个元素的地址
    row_clk_offset  = []    # 第r行有效运行的时间, 如果前面是空拍则为负
    row_ofmap_idx   = []    # 第r行计算的一个channel ofmap的px标号, 从0~e2-1
    v_fold_row      = []    # 这一列已经计算了几个channel的output
    col_base_addr   = []    # 每列的filter基地址
    col_clk_offset  = []    
    v_fold_col      = []    # 第r行处理到了第几个channel的ofmap, 从0~num_v_fold
    h_fold_col      = []
    lane_done       = []    #
    v_fold_barrier  = []    # 该行是否需要barrier直到这张图被计算完才开始下一次计算

    # Variables for utilization calculation
    rows_used = 0
    cols_used = 0
    util      = 0


    # This initialization assumes num_rows << num_ofmap_px
    # The assignment logic needs to be modified if that is not the case
    for r in range(dim_rows):
        # 沿着列输入的是ifmap, 如果r > ofmap_w, 意味着卷积向下滑动了stride
        base_row_id = math.floor(r / ofmap_w) * stride
        # 进入阵列每行第一个数的地址,沿着行输入的是ifmap, 假设r<ofmap_w, 那么就相当于第一行1, 第二行1+stride...
        base_col_id = r % ofmap_w * stride
        # 每一行
        base_addr = base_row_id * hc + base_col_id * num_channels
        # print((base_row_id, base_col_id, base_addr)) 

        if r < e2:
            clk_offset = r * -1             # Clock offset takes care of the skew due to store and forward
        else:
            clk_offset = neg_inf            # In case num_ofamp_px < dim_rows, 初始化为负无穷

        
        row_base_addr.append(base_addr)     # 每行的ifmap第一个元素的地址
        row_clk_offset.append(clk_offset)   # 每行的skew导致的延时
        row_ofmap_idx.append(r)             # 每行计算的ofmap index标号
        v_fold_row.append(0)                #
        v_fold_barrier.append(False)        #

    for c in range(dim_cols):
        # 这里的是每一列filter的基地址, 因为沿着col展开的是各个卷积核
        base_addr = c * r2c

        # Anand: TODO
        if c < remaining_filt:
            clk_offset = c * -1
            lane_done.append(False)
        else:
            clk_offset = neg_inf
            lane_done.append(True)          #lane done是指是否所有filter被加载完

        col_base_addr.append(base_addr)
        col_clk_offset.append(clk_offset)   #每一列都延后一拍
        v_fold_col.append(0)
        h_fold_col.append(0)


    # Open tracefile for writing
    outfile     = open(sram_read_trace_file, 'w')
    #ofmap_out   = open(sram_write_trace_file, 'w')

    # Adding progress bar
    tot  = e2 * num_v_fold
    #print("Total = " + str(tot))
    pbar = tqdm(total=tot)

    # Generate traces here
    # The condition checks
    #       1)  if the all the input traces for last v fold is generated
    # and   2)  if all the filter traces have been generated
    #while(remaining_px[num_v_fold-1] > 0) or (filt_done == False):
    while(ifmap_done == False) or (filt_done == False):
        ifmap_read = ""
        filt_read  = ""
        rows_used = 0
        cols_used = 0
        
        # Generate address for ifmap
        for r in range(dim_rows):

            if(row_clk_offset[r] >= 0):     # Take care of the skew

                inc = row_clk_offset[r]

                # 这个地址理解的不是很好, 先放着, 但是可以肯定的是ifmap的
                addr_row_offset = math.floor(inc / rc) * ifmap_w * num_channels
                addr_col_offset = inc % rc
                ifmap_addr = row_base_addr[r] + addr_row_offset + addr_col_offset 
                ifmap_read += str(int(ifmap_addr)) + ", "
                rows_used += 1
            else:
                ifmap_read += ", "

            # 消耗前面的空拍
            row_clk_offset[r] += 1

            if (row_clk_offset[r] > 0) and (row_clk_offset[r] % r2c == 0):   #Completed MAC for one OFMAP px
                
                # 表示已经计算完一圈了, 所以每行计算的标号要增加dim_rows
                #row    index
                #0      0       3   (r2c个cycle计算完), idx更新为0+3
                #1      1       4   (r2c个cycle计算完), idx更新为1+3, 注意这一行慢一拍
                #2      2       5   ....
                row_ofmap_idx[r] += dim_rows  
                ofmap_idx = row_ofmap_idx[r]

                # Update progress bar
                pbar.update(1) 

                # 判断是否已经计算完了一个ofmap channel
                if ofmap_idx < e2:
                    row_clk_offset[r] = 0

                    base_row_id = math.floor(ofmap_idx / ofmap_w) * stride
                    base_col_id = ofmap_idx % ofmap_w * stride
                    base_addr  = base_row_id * hc + base_col_id * num_channels
                    row_base_addr[r] = base_addr

                else:
                    v_fold_row[r] += 1
                    #pbar.update(e2)

                    if (v_fold_row[r] < num_v_fold):
                        # 计算完了一个ofmap channel, 所以ofmap idx要更新
                        # 这里的v_fold_row是oc/nums-cols, 也是每一列要计算的Toc数量
                        row_ofmap_idx[r]  = r

                        base_row_id = math.floor(r / ofmap_w) * stride
                        base_col_id = r % ofmap_w * stride
                        base_addr  = base_row_id * hc + base_col_id * num_channels
                        row_base_addr[r]  = base_addr

                        # Stall this col from proceeding until all the rows reach the v_fold boundary
                        # 新的一行已经计算到了新的oc, 但是这一行上面的行还没有计算到, 需要设置一个barrier, 
                        # barrier[r] == true的时候, 不能进行计算
                        # 阵列形状对bubble的影响非常大, 当e2/col的余数越小时, 最终的bubble越小, 当其能完全整除时, 没有bubble
                        # 这一点可以改变阵列的形状进来验证
                        '''
                        (row idx barrier) idx barrier ....
                        (0	32	 False)	32	 False)	0	 False)	0	 False)	0	 False)	0	 False)	0	 False)	8	 False)	8
                        (1	33	 False)	33	 False)	33	 False)	1	 False)	1	 False)	1	 False)	1	 False)	1	 False)	9
                        (2	34	 False)	34	 False)	34	 False)	34	 False)	2	 False)	2	 False)	2	 False)	2	 False)	2
                        (3	27	 False)	35	 False)	35	 False)	35	 False)	35	 False)	3	 False)	3	 False)	3	 False)	3
                        (4	28	 False)	28	 False)	4	 True)	4	 True)	4	 True)	4	 True)	4	 True)	4	 False)	4
                        (5	29	 False)	29	 False)	29	 False)	5	 True)	5	 True)	5	 True)	5	 True)	5	 True)	5
                        (6	30	 False)	30	 False)	30	 False)	30	 False)	6	 True)	6	 True)	6	 True)	6	 True)	6
                        (7	23	 False)	31	 False)	31	 False)	31	 False)	31	 False)	7	 True)	7	 True)	7	 True)	7
                        '''
                        if (r != 0) and ((v_fold_row[r] > v_fold_row[r-1]) or (v_fold_barrier[r-1] == True)):
                            row_clk_offset[r] = neg_inf
                            v_fold_barrier[r] = True
                        else:
                            row_clk_offset[r] = 0

                    else:
                        row_clk_offset[r] = neg_inf
            # print((r, row_ofmap_idx[r], v_fold_barrier[r], v_fold_row[r]))

        # Get out of the barrier one by one
        # IMPORTANT: The barrier insertion and recovery is in separate loops to ensure that
        #            in a given clock cycle insertion for all rows strictly happen before the release.
        #            The flag ensures only one col is released per cycle
        # Since indx 0 never enters the barrier, this should work fine
        flag = False
        for r in range(dim_rows):
            # 这一部分在release barrier
            if v_fold_barrier[r] and flag==False:
                if (v_fold_row[r] == v_fold_row[r-1]) and (v_fold_barrier[r-1] == False):
                    v_fold_barrier[r] = False
                    flag = True
                    # 因为之前stall住了, 要对clk进行修正
                    row_clk_offset[r] = row_clk_offset[r-1] -1

        # Check if all input traces are done
        ifmap_done = True
        for r in range(dim_rows):
            if row_clk_offset[r] > 0:
                ifmap_done = False

        # Generate address for filters
        for c in range(dim_cols):
            if(col_clk_offset[c] >= 0):     # Take care of the skew
                inc = col_clk_offset[c]
                
                filt_addr = col_base_addr[c] + inc + filt_base 
                filt_read += str(filt_addr) + ", "
                cols_used += 1
            else:
                filt_read += ", "

            col_clk_offset[c] += 1

            # 列已经
            if(col_clk_offset[c] > 0) and (col_clk_offset[c] % r2c == 0):

                # Get the v fold this col is working on and check the status of input trace generation
                #rem_px = remaining_px[v_fold_col[c]]

                #Tracking on the basis of h_folds
                h_fold_col[c] += 1

                # Anand: Check if all the input traces are generated for the given v fold
                # 每一列执行到了第几个个Toc
                if (h_fold_col[c] < num_h_fold):
                    col_clk_offset[c] = 0
                else:
                    v_fold_col[c] += 1
                    filt_id = v_fold_col[c] * dim_cols + c

                    # All filters might not be active in the last fold
                    # Adding the filter ID check to ensure only valid cols are active
                    if(v_fold_col[c] < num_v_fold) and (filt_id < num_filters):
                        col_clk_offset[c] = 0
                        h_fold_col[c] = 0

                        base = filt_id * r2c
                        col_base_addr[c] = base

                    else:
                        col_clk_offset[c] = neg_inf
                        lane_done[c] = True

        # Check if all filter traces are generated
        filt_done = True
        for c in range(dim_cols):
            # 单个车道是否计算完毕了, 这里的车道是指一列
            if lane_done[c] == False:
                filt_done = False

                                                
        # Write to trace file
        global_cycle = cycle + local_cycle
        entry = str(global_cycle) + ", " + ifmap_read + filt_read + "\n"
        outfile.write(entry)

        this_util = (rows_used * cols_used) / (dim_rows * dim_cols)
        util += this_util

        # Update tracking variables
        local_cycle += 1
        # print(local_cycle, filt_done, ifmap_done)
    pbar.close()
    outfile.close()
    #ofmap_out.close()

    util_perc = (util / local_cycle) * 100

    return (local_cycle + cycle), util_perc
# End of gen_read_trace()


def gen_write_trace(
        cycle = 0,
        dim_rows = 4,
        dim_cols = 4,
        #num_v_fold = 1,
        #num_h_fold = 1,
        ofmap_h = 5, ofmap_w = 5,
        num_filters = 4,
        ofmap_base = 2000000,
        conv_window_size = 9,                      # The number of pixels in a convolution window
        sram_write_trace_file = "sram_write.csv"
):

    # Layer specific variables
    r2c = conv_window_size
    e2  = ofmap_h * ofmap_w

    # Tracking variables
    id_row = []             # List of OFMAP ID for each row
    id_col = []             # List of filter ID for each col
    base_addr_col =[]       # Starting address of each output channel
    remaining_px  = e2
    remaining_filt= num_filters
    active_row = min(dim_rows, e2)
    active_col = min(dim_cols, num_filters)
    local_cycle = 0
    sticky_flag = False     # This flag is in place to fix the OFMAP cycle shaving bug


    for r in range(active_row):
        id_row.append(r)

    for c in range(active_col):
        id_col.append(c)

        base_col = c
        base_addr_col.append(base_col)

    #Open the file for writing
    outfile = open(sram_write_trace_file,"w")

    #This is the cycle when all the OFMAP elements in the first col become available
    #第一列元素计算完的时间为
    local_cycle = r2c + active_col - 1

    # 还有剩余px或者filter没有被计算时
    while (remaining_px > 0) or (remaining_filt > 0):

        # 每行都是一个pixels
        active_row = min(dim_rows, remaining_px)

        for r in range(active_row):
            local_px = id_row[r]
            remaining_px -= 1
            id_row[r] += active_row     # Taking care of horizontal fold

            ofmap_trace = ""
            for c in range(active_col):
                addr = ofmap_base + base_addr_col[c] + local_px * num_filters
                ofmap_trace += str(addr) + ", "

            # Write the generated traces to the file
            entry = str(local_cycle + r) + ", " + ofmap_trace + "\n"
            outfile.write(entry)

        # Take care of the vertical fold
        if remaining_px == 0:
            remaining_filt -= active_col

            # In case of vertical fold we have to track when the output of (0,0) is generated
            # Shifting back local cycles to capture the last OFMAP generation in (0,0) for this fold
            last_fold_cycle   = local_cycle + active_row
            # local_cycle -= (active_row + active_col - 1)
            # 这里写错了
            local_cycle -= (active_col - 1)
            sticky_flag = True

            # There are more OFMAP channels to go
            if remaining_filt > 0:
                remaining_px = e2
                last_active_col = active_col
                active_col = min(remaining_filt, dim_cols)

                # Reassign col base addresses
                for c in range(active_col):
                    base_addr_col[c] += last_active_col

                active_row = min(dim_rows, remaining_px)
                # Reassign row base addresses
                for r in range(active_row):
                    id_row[r] = r

                local_cycle += r2c + active_col
                if local_cycle < last_fold_cycle:
                    local_cycle = last_fold_cycle


            else:   # Restore the local cycle to return to the main function
                local_cycle = last_fold_cycle
                #local_cycle += (active_row + active_col)
                #sticky_flag = False

        else:   # If this is not a vertical fold then it is business as usual
            local_cycle += max(r2c, active_row)

    outfile.close()

    #if sticky_flag:
    #    local_cycle += (active_row + active_col)
    #    sticky_flag = False

    return(local_cycle + cycle)
# End of gen_write_trace()


if __name__ == "__main__":
   sram_traffic(
       dimension_rows = 4,
       dimension_cols = 4,
       ifmap_h = 7, ifmap_w = 7,
       filt_h = 2, filt_w = 2,
       num_channels = 1, strides = 1,
       num_filt = 16
   )
