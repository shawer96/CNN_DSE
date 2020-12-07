import math 
from tqdm import tqdm




def sram_traffic(
        dimension_rows=4,                   #    阵列的行
        dimension_cols=4,                   #    阵列的高   
        ifmap_h=7, ifmap_w=7,               #    ifmap的高和宽
        filt_h=3, filt_w=3,                 #    filter的高和宽
        num_channels=3,                     #    ifmap和filter的通道数
        strides=1, num_filt=8,              #    filter的个数，output的通道数
        ofmap_base=2000000, filt_base=1000000, ifmap_base=0,    #
        sram_read_trace_file="sram_read.csv",
        sram_write_trace_file="sram_write.csv"
    ):
    # 输入为
    '''
    阵列的行
    阵列的高   
    ifmap的高和宽
    filter的高和宽
    ifmap和filter的通道数
    filter的个数，output的通道数
    '''
    # Dimensions of output feature map channel
    E_h = math.floor((ifmap_h - filt_h + strides) / strides)    #ofmap的高
    E_w = math.floor((ifmap_w - filt_w + strides) / strides)    #ofmap的宽
    
    # Number of pixels in one convolution window
    px_per_conv_window = filt_h * filt_w * num_channels     #单个卷积核的大小（单位为pixels）
    r2c = px_per_conv_window                                #单个卷积核的大小（单位为pixels）

    # Total number of ofmap px across all channels
    num_ofmap_px = E_h * E_w * num_filt                     #单层output的大小（单位为pixels）
    e2  = E_h * E_w                                         #单个output channel的大小（单位为pixels）
    e2m = num_ofmap_px                                      #单层output的大小（单位为pixels）
    
    '''
    '''
    # Variables to calculate folds in runtime
    num_h_fold = 1  #初始化垂直方向折叠（分块）次数（循环多少次）
    num_v_fold = 1  #初始化水平方向折叠（分块）次数（循环多少次）
    max_parallel_window = 1 #初始化每个pe col并行的卷积窗数量

    # Variables for utilization calculation
    util = 0
    compute_cycles = 0

    '''
    因为是要把单个卷积核映射到阵列的col上，
    所以要比较单个卷积核和col的大小
    if col上的计算单元数量比单个卷积核的像素少：那么要折叠单个卷积核
    else col上的计算单元数量比单个卷积核的像素多：那么要展开多个卷积核max_parallel_window次
    '''
    if dimension_rows < px_per_conv_window:
        num_h_fold = math.ceil(px_per_conv_window/dimension_rows)
    else:
        max_parallel_window = math.floor(dimension_rows/ px_per_conv_window)

    '''
    计算同时能计算多少个卷积窗,也就是在oc维度上要分块多少次
    '''
    reqd_cols = num_filt                    # Total number of cols to be mapped也就是Noc的数量
    max_cols_per_v_fold = max_parallel_window * dimension_cols  #同时有多少个卷积窗
    num_v_folds = math.ceil(reqd_cols / max_cols_per_v_fold)    #
    
    remaining_cols = reqd_cols          #因为通常不能整除,实际最后一次其实是不规则的, 有remaining
    cycles = 0
    prev_cycl = 0

    #print("Vertical folds = " +str(num_v_folds))
   
    # These are the starting addresses of filter weights in the memory
    all_col_addr_list = []  #这里存储的是每个卷积核的基地址
    for c in range(num_filt):
        addr = (c) * r2c + filt_base 
        all_col_addr_list.append(addr)

    # These are the starting addresses of ifmap windows in the memory
    hc = ifmap_w * num_channels #这里存储的是每个ifmap channel的基地址
    all_ifmap_base_addr = []
    for px in range(int(e2)):         #number of ofmap px in a ofmap channel
        addr = (px / E_w) * strides * hc + (px%E_w) * strides
        all_ifmap_base_addr.append(addr)

    #tdqm为进度条模块
    #沿着oc方向遍历每个Toc, 这里的v相当于是块的标号, 这里是先遍历oc再ic, 相当于是优先输出的
    for v in tqdm(range(int(num_v_folds))):
        #print("V fold id: " + str(v))
            
        # Take a slice of the starting addresses that are relevant for this v_fold 

        # 本次循环要遍历的col数量, 因为是最后一次其实会有剩余
        cols_this_fold = min(remaining_cols, max_parallel_window * dimension_cols)
        idx_start = v * dimension_cols          #每一块的起始地址
        idx_end = idx_start + cols_this_fold    #每一块的结束地址
        col_addr_list = all_col_addr_list[idx_start:idx_end]    #weight块地址的排布

        # 这里开始遍历每个循环, 这里的if对应的是算力小于单个卷积核尺寸的情况
        if num_h_fold > 1 :
           
            # 单个卷积核的像素数量
            rem_h = r2c                     # Tracks the elements processed within a conv filter 
            next_ifmap_addr = ifmap_base    # Starts from the top left corner of the IFMAP matrix

            #这里相当于是在遍历整个每个列
            for h in range(num_h_fold):
                # 遍历一列的元素, 是剩余元素和阵列高度中较小的一个
                rows_this_fold = min(rem_h, dimension_rows) 
                #print("h fold id: " + str(h))

                # Values returned
                # cycles        -> Cycle count for the next operation ie. cycles elapsed + 1
                # col_addr_list -> The starting filter address for the next iteration
                # 计算预载weight的时间
                cycles, col_addr_list   = gen_trace_filter_partial(
                                            col_addrs   = col_addr_list,
                                            cycle       = cycles,
                                            num_rows    = dimension_rows,
                                            remaining   = rows_this_fold,
                                            sram_read_trace_file = sram_read_trace_file
                                            )
                # print("Weights loaded by " + str(cycles) + " cycles")
                # ifmap和ofmap的cycle都是从weight load结束之后才开始计算
                data_out_cycles     = cycles    #Store this cycle for parallel readout
                cycles_ifmap            = gen_trace_ifmap_partial(
                                            cycle = cycles,
                                            num_rows = dimension_rows, num_cols = dimension_cols,
                                            num_filters = num_filt,
                                            remaining = rem_h,
                                            remaining_filters = remaining_cols, 
                                            ifmap_h = ifmap_h, ifmap_w = ifmap_w,
                                            filt_h = filt_h, filt_w = filt_w,
                                            num_channels = num_channels,
                                            stride = strides, ifmap_base = ifmap_base,
                                            sram_read_trace_file = sram_read_trace_file
                                            )
                cycles_ofmap        = gen_trace_ofmap(
                                            cycle = data_out_cycles,
                                            num_rows = dimension_rows,
                                            num_cols = dimension_cols,
                                            ofmap_base = ofmap_base,
                                            window_size= rows_this_fold,
                                            parallel_window =1,
                                            num_ofmap_px = int(e2),
                                            filters_done = (v * dimension_cols),
                                            num_filter = num_filt,
                                            sram_write_trace_file = sram_write_trace_file
                                            ) 

                #print("IFMAPS processed by " + str(cycles) + " cycles")
                # 参与计算的PE数量占总计算资源的比例, 这里是每个cols_this_fold的计算量
                util_this_fold = (rows_this_fold * cols_this_fold) /(dimension_rows * dimension_cols)

                rem_h -= rows_this_fold  # 一个卷积核剩余的元素数量
                # ifmap的load和ofmap的store同时发生, 是相互隐藏的
                cycles = max(cycles_ifmap, cycles_ofmap)

                # 执行this_fold所花费的时间
                del_cycl = cycles - prev_cycl
                util += util_this_fold *  del_cycl
                compute_cycles += del_cycl
                prev_cycl = cycles

        # 下面是计算阵列的大小大于卷积窗时的函数
        else:
            #filters_this_fold = min(remaining_cols, max_cols_per_v_fold)
            filt_done = v * max_parallel_window * dimension_cols
            rem = num_filt - filt_done

            parallel_window = math.ceil(rem / dimension_cols)
            parallel_window = int(min(max_parallel_window, parallel_window))
        
            # 这个函数主要是在计算读取权重的cycle数
            cycles_filter = gen_filter_trace(
                                cycle = cycles,
                                num_rows = dimension_rows, num_cols = dimension_cols,
                                filt_h = filt_h, filt_w = filt_w, num_channels = num_channels,
                                col_addr = col_addr_list, 
                                parallel_window=parallel_window,
                                filters_this_fold=cols_this_fold,
                                sram_read_trace_file=sram_read_trace_file
                                )
            # 这一部分是load ifmap的时间
            cycles_ifmap, rows_this_fold\
                            = gen_ifmap_trace(
                            cycle = cycles_filter,
                            num_rows = dimension_rows, num_cols = dimension_cols,
                            ifmap_h = ifmap_h, ifmap_w = ifmap_w,
                            filt_h = filt_h, filt_w = filt_w,
                            num_channels = num_channels, stride = strides,
                            parallel_window = parallel_window,
                            sram_read_trace_file = sram_read_trace_file
                            )

            # 这一部分是load
            cycles_ofmap = gen_trace_ofmap(
                            cycle = cycles_filter,
                            num_rows = dimension_rows, num_cols = dimension_cols,
                            ofmap_base = ofmap_base, 
                            parallel_window = parallel_window,
                            window_size = r2c,
                            num_ofmap_px = int(e2),
                            filters_done = int(v * max_parallel_window * dimension_cols),
                            num_filter = num_filt,
                            sram_write_trace_file = sram_write_trace_file
                            )
            cycles = max(cycles_ifmap, cycles_ofmap)
            del_cycl = cycles - prev_cycl

            # Since multiple filters are being mapped on a single col due to large number of rows
            # util calculation is a little involved,
            # cols_this_fold --> number of filters mapped this fold
            rem = cols_this_fold    
            tmp_util = 0
            for _ in range(parallel_window):
                #因为这里的rem包含了col_this_fold,也就包含了一列的展开数, 所以这里的row_used就是一个filter的大小
                col_used = min(rem, dimension_cols) 
                row_used = r2c                      # Number of row used will always be in multiple of r2c,
                                                    # parallel window calc took care of this
                tmp_util += row_used * col_used
                rem -= col_used

            #util_this_fold = (rows_this_fold * cols_this_fold) /(dimension_rows * dimension_cols)
            util_this_fold = tmp_util /(dimension_rows * dimension_cols)
            util += util_this_fold * del_cycl
            compute_cycles += del_cycl
            prev_cycl = cycles

        remaining_cols -= cols_this_fold

    final = str(cycles)
    final_util = (util / compute_cycles) * 100
    #print("Compute finished at: " + str(final) + " cycles")
    return (final, final_util)


def gen_filter_trace(
        cycle = 0,
        num_rows = 4, num_cols = 4,
        filt_h = 3, filt_w = 3, num_channels = 3,
        col_addr = [],
        parallel_window = 1,
        filters_this_fold = 4,
        sram_read_trace_file = "sram_read.csv"
):
    outfile = open(sram_read_trace_file,'a')
 
    # There is no data from the left side till the weights are fed in
    # This prefix is to mark the blanks
    prefix  = ""
    for r in range(num_rows):
        prefix += ", "

    # Calculate the convolution window size
    r2c = filt_h * filt_w * num_channels 

    rem = filters_this_fold                 # Track the number of filters yet to process

    #For each wrap around
    for w in range(parallel_window):
        # Number of active columns in this wrap
        cols = min(num_cols, rem)
        rem -= cols

        # For each row in the window
        for r in range(r2c):
            entry = str(cycle) + ", " + prefix
            cycle += 1
            
            # In each cycle, for each column feed one weight
            for c in range(cols):
                indx  = w * num_cols + c
                entry += str(col_addr[indx]) + ", "         
                col_addr[indx] += 1

            if cols < num_cols:
                for _ in range(c, num_cols):
                    entry += ", "

            entry += "\n"
            outfile.write(entry)
 
    outfile.close()
    return cycle


def gen_ifmap_trace(
        cycle = 0,
        num_rows = 4, num_cols = 4,
        ifmap_h = 7, ifmap_w = 7,
        filt_h = 3, filt_w = 3,
        num_channels = 3, stride = 1,
        parallel_window = 1,
        sram_read_trace_file = "sram_read.csv"
):
    outfile = open(sram_read_trace_file,'a')
    postfix = ""
    for c in range(num_cols):
        postfix += ", "
    
    E_h = math.floor((ifmap_h - filt_h + stride) / stride)
    E_w = math.floor((ifmap_w - filt_w + stride) / stride)
    e2  = E_h * E_w
    r2c = filt_h * filt_w * num_channels
    rc = filt_w * num_channels
    hc = ifmap_w * num_channels

    idle = num_rows - (r2c * parallel_window)
    idle = max(idle, 0)
    used_rows = num_rows - idle

    # Adding entries for columns and empty rows
    #print("Idle lanes = " + str(idle))
    idle += num_cols
    for i in range(idle):
        postfix += ", "
    postfix += "\n"

    base_addr = 0
    
    for e in range(int(e2)):
        entry = str(cycle) + ", "
        cycle += 1    

        #print("Cycle= " + str(cycle))
        #Inner loop for all the rows in array
        num_rows = r2c 
        row_entry = []
        for r in range(num_rows):
            row_idx = math.floor(r / rc)  # math.floor to get in integral value
            col_idx = r % rc 
            add = base_addr + row_idx * hc + col_idx 
            #print("Row idx " + str(row_idx) + " col_idx " + str(col_idx) +" add " + str(add))
            row_entry.append(add)

        # Reverse the printing order
        # Reversal is needed because the filter are stored in upside down order in the array
        # ie. last row has the first weight element
        l = len(row_entry)
        #print("Parallel windows = " + str(parallel_window))
        for w in range(parallel_window):
            #print("Window = " + str(w))
            for ridx in range(l):
                entry += str(row_entry[l - ridx -1]) + ", "

        entry += postfix
        outfile.write(entry)

        # Calculate the IFMAP addresses for next cycle
        px_this_row = (e+1) % E_w
        if px_this_row == 0:
            #print("New row")
            ifmap_row = math.floor(base_addr / hc)
            base_addr = (ifmap_row +  stride) * hc
        else:
            base_addr += stride * num_channels
        #print("OFAMP px = " + str(e+1) + " base_addr: " + str(base_addr))

    outfile.close()
    return cycle, used_rows


def gen_trace_filter_partial(
                    col_addrs=[],       #Ensure that this takes care of the v_folding
                    cycle=0,
                    num_rows=4,
                    remaining=4,
                    sram_read_trace_file="sram_read.csv"
):
        # 记录从sram读取weight到PEA的过程
        outfile = open(sram_read_trace_file, 'a')
        num_cols = len(col_addrs)

        # output formatting: Add empty commas for row addresses as no element is fed from the left
        prefix = ""
        for r in range(num_rows):
            prefix += ", "

        # Entries per cycle 
        # 预载weight, 时间与此次预载每一列中的weight数量有关
        # weight预载的时间就是remaining的数量
        for r in range(remaining):              # number of rows this cycle
            entry = str(cycle) + ", " + prefix

            for c in range(num_cols):
                entry += str(col_addrs[c]) + ", "
                col_addrs[c] += 1
            
            cycle += 1
            entry += "\n"
            outfile.write(entry)

        outfile.close()

        return cycle, col_addrs 


def gen_trace_ifmap_partial(
                    cycle = 0,
                    num_rows = 4, num_cols = 4,
                    remaining=4,
                    num_filters = 8,            #   
                    remaining_filters = 0,      # These two are used to track the reads of PS
                    ifmap_h = 4, ifmap_w = 4,
                    filt_h = 3, filt_w = 3,
                    num_channels = 3,
                    stride = 1, 
                    ifmap_base = 0, ofmap_base = 2000000,
                    sram_read_trace_file = "sram_read.csv"
):
    outfile = open(sram_read_trace_file, 'a')
    postfix = ""
    for c in range(num_cols):
        postfix += ", "
    postfix += "\n"

    r2c = filt_h * filt_w * num_channels
    rc = filt_w * num_channels
    hc = ifmap_w * num_channels
    E_w = (ifmap_w - filt_w + stride) / stride 
    E_h = (ifmap_h - filt_h + stride) / stride 

    num_ofmap_px = E_h * E_w
    index = r2c - remaining
    base_addr = 0 
            
    filter_done = num_filters - remaining_filters
    #outfile.write(str(filter_done) + ", " + str(num_filters)+", "+str(remaining_filters)+", "+ "\n")
    #ofmap_offset = filter_done * num_ofmap_px
    ofmap_offset = filter_done
    # 有效计算的cols数量
    effective_cols = min(remaining_filters, num_cols)
    tick = 0                                # Proxy for clock to track input skewing

    # Outerloop for all ofmap pixels in an ofmap channel
    # 每一列计算output pixels的部分和, 对于每一个output pixels, 都需要一个cycle
    for e in range(int(num_ofmap_px)):
        entry = str(cycle) + ", "
        cycle += 1    

        #print("Cycle= " + str(cycle))
        #Inner loop for all the rows in array
        num_rows = min(num_rows, remaining)
        row_entry = []
        for r in range(num_rows):
            row_idx = math.floor((index+r) / rc)  # math.floor to get in integral value
            col_idx = (index+r) % rc 
            add = base_addr + row_idx * hc + col_idx 
            #print("Row idx " + str(row_idx) + " col_idx " + str(col_idx) +" add " + str(add))
            row_entry.append(add)

        # Reverse the printing order
        # Reversal is needed because the filter are stored in upside down order in the array
        # ie. last row has the first weight element
        l = len(row_entry)
        for ridx in range(l):
            entry += str(row_entry[l - ridx -1]) + ", "

        # In case of partial mapping
        # index > 0 implies that there is a partial sum generated from prev h_fold
        # This partial sum is now fed from the top to be summed with the PS generated in this h_fold
        # The following part print the read addresses for PS
        # Anand : TODO, Implementation choice, do not support right now
        '''
        if index > 0:
            postfix = ""
            for c in range(effective_cols):
                if (tick - c) > -1:                       # Track PS reads for skew
                    a = (e - c) * num_filters + c        # e - c: Taking care of skew by c cycles
                    a = a + ofmap_base + ofmap_offset
                    postfix += str(a) + ", "
                else:
                    postfix += ", "
            tick += 1
            #print("Tick =", str(tick) + "Postfix= " + postfix)
            postfix += "\n"
        '''
        entry += postfix
        outfile.write(entry)

        px_this_row = (e+1) % E_w
        if px_this_row == 0:
            #print("New row")
            ifmap_row = math.floor(base_addr / hc)
            base_addr = (ifmap_row + stride) * hc
        else:
            base_addr += stride * num_channels
        #print("OFAMP px = " + str(e+1) + " base_addr: " + str(base_addr))

    outfile.close()
    return cycle


def gen_trace_ofmap(
                    cycle = 0,
                    num_rows = 4, num_cols =4,
                    ofmap_base = 2000000,
                    parallel_window = 1,
                    window_size = 27,
                    num_ofmap_px = 16,      # This is per ofmap channel
                    filters_done = 0,       # To track v fold
                    num_filter   = 8,       # To track if all filters have finished
                    sram_write_trace_file = "sram_write.csv"
):
    outfile = open(sram_write_trace_file,'a')
    #cycle = num_cols + cycle     # Accounts for the time taken to reduce accross all cols

    # Corner case when parallel_window = 1, but num_filter < num_cols
    if parallel_window > 1:
        cycle += num_cols
        cycle += window_size                # window_size == r2c
    else:
        rem    = (num_filter - filters_done)
        cycle += min(rem, num_cols)
        cycle += window_size

    #ofmap_add_offset  = filters_done * num_ofmap_px
    ofmap_add_offset  = filters_done
    remaining_filters = num_filter - filters_done
    
    effective_cols    = num_cols * parallel_window
    effective_cols    = min(effective_cols, remaining_filters)

    # 考虑单个oc, weight需要被复用Nox*Noy*Noc次才能计算出单个output pixels
    for e in range(int(num_ofmap_px)):
        entry = str(cycle) + ", "
        cycle += 1
        
        done = filters_done
        for col in range(effective_cols):
            if done < num_filter:
                a = e * num_filter + col                # z first row major
                a = a + ofmap_add_offset + ofmap_base
                entry += str(a) + ", "
            else: 
                # Code should not enter this part
                entry += "!, "

        entry += "\n"
        outfile.write(entry)

    outfile.close()
    return cycle


# Trace generation for moving generated ofmap data in cases when only partial window fits
# This implementation prints out the ofmap pixel in the exact cycle it is generated
# Not used in scale sim at the moment. 
# SCALE sim waits till all the columns finish generating OFMAP.
def gen_trace_ofmap_partial_imm(
                        cycle = 0,
                        num_rows = 4, num_cols =4,
                        ofmap_base = 2000000,
                        num_ofmap_px = 16,
                        num_filter = 8,
                        filters_done = 0,
                        sram_write_trace_file = "sram_write.csv"
):
    outfile = open(sram_write_trace_file,'a')
    start_cycle = num_rows + cycle

    col_addr = []
    for col in range(int(num_cols)):
        a = (filters_done + col)
        col_addr.append(a)
    
    #output cycles d 
    for tick in range(int(num_ofmap_px + num_cols)):
        cycle = start_cycle + tick

        entry = str(cycle) + ", "
        for col in range(int(num_cols)):
            # Condition to maintain skew
            if tick >= col and (tick - col)< num_ofmap_px:
                entry += str(col_addr[col]) + ", "
                col_addr[col] += num_filter
            else:
                entry += ", "
        
        entry += "\n"
        outfile.write(entry)

    outfile.close()


if __name__ == "__main__":
    h_h = 5 
    h_w = 5

    r_h = 2
    r_w = 2

    c = 2
    u =1

    m = 9

    dim_h = 16
    dim_v = 5

    sram_traffic(
        dimension_rows = dim_h,
        dimension_cols = dim_v,

        ifmap_h = h_h, ifmap_w = h_w,
        filt_h = r_h, filt_w = r_w, 
        num_channels = c,
        strides = u,

        num_filt = m
    )
