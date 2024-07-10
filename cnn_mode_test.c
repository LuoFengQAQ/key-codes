/*
 * Copyright (c) 2022 HiSilicon (Shanghai) Technologies CO., LIMITED.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <sys/prctl.h>
#include <math.h>

// #include "hi_comm_svp.h"
// #include "hi_nnie.h"
// #include "mpi_nnie.h"

#include "hi_common.h"
#include "hi_comm_sys.h"
#include "hi_comm_svp.h"
#include "sample_comm.h"
#include "sample_comm_svp.h"
#include "sample_comm_nnie.h"
#include "sample_nnie_main.h"
#include "sample_svp_nnie_software.h"
#include "sample_comm_ive.h"
#include "cnn_mode_test.h"

/*主要参考了示例中分割网部分的代码*/
void cnn_mode_test(void)
{
    /*HI_CHAR path[PATH_MAX] = {0};相关定义，属于字符数组*/
    /*这个输入，当时在pycharm那边改成了(1,2,1,30)的格式，文件中是按x1 y1 x2 y2...这种排的，一条轨迹的数据放到一行，已经做了归一化*/
    /*数据格式是主要是根据量化时的规定来的，如果直接在程序里面输入，应该也是这种格式*/
    const HI_CHAR *pcSrcFile = "./data/nnie_image/track/track_history.txt";
    const HI_CHAR *pcModelName = "./data/nnie_model/prediction/cnn_convet_inst.wk";
    const HI_U32 u32PicNum = 1;
    HI_S32 s32Ret = HI_SUCCESS;
    SAMPLE_SVP_NNIE_CFG_S stNnieCfg = { 0 };
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S stInputDataIdx = { 0 };
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S stProcSegIdx = { 0 };

    /* Set configuration parameter */
    /*模型的一些参数，pszPic是指输入，本网络中就是对应需要预测的历史航迹*/
    stNnieCfg.pszPic = pcSrcFile;
    stNnieCfg.u32MaxInputNum = u32PicNum; // max input image num in each batch
    stNnieCfg.u32MaxRoiNum = 0;
    stNnieCfg.aenNnieCoreId[0] = SVP_NNIE_ID_0;
    g_stop_signal = HI_FALSE;

    /* Sys init */
    s32Ret = SAMPLE_COMM_SVP_CheckSysInit();
    SAMPLE_SVP_CHECK_EXPR_RET_VOID(s32Ret != HI_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_COMM_SVP_CheckSysInit failed!\n");
    if (g_stop_signal == HI_TRUE) {
        SAMPLE_SVP_NNIE_Segnet_Stop();
        return;
    }
    /* Segnet Load model */
    SAMPLE_SVP_TRACE_INFO("Segnet Load model!\n");
    /*s32Ret接受的是函数的返回值，是表示是否读取成功的标志0或1，并不是读取的模型本身，后面的函数应该也都是这种模式*/
    s32Ret = SAMPLE_COMM_SVP_NNIE_LoadModel(pcModelName, &s_stSegnetModel);//读取模型结构到了s_stSegnetModel中
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, SEGNET_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_COMM_SVP_NNIE_LoadModel failed!\n");

    /* Segnet parameter initialization */
    SAMPLE_SVP_TRACE_INFO("Segnet parameter initialization!\n");
    if (g_stop_signal == HI_TRUE) {
        SAMPLE_SVP_NNIE_Segnet_Stop();
        return;
    }
    s_stSegnetNnieParam.pstModel = &s_stSegnetModel.stModel;
    /*对上述读取的模型进行初始化，stNnieCfg即预先设定的参数，s_stSegnetNnieParam应该是模型结构中的权重之类的*/
    s32Ret = SAMPLE_SVP_NNIE_Cnn_ParamInit(&stNnieCfg, &s_stSegnetNnieParam, NULL);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, SEGNET_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Cnn_ParamInit failed!\n");

    /* Fill src data */
    SAMPLE_SVP_TRACE_INFO("Segnet start!\n");
    stInputDataIdx.u32SegIdx = 0;
    stInputDataIdx.u32NodeIdx = 0;
    if (g_stop_signal == HI_TRUE) {
        SAMPLE_SVP_NNIE_Segnet_Stop();
    }
    /*应该是将要进行预测的数据输入到模型中*/
    s32Ret = SAMPLE_SVP_NNIE_FillSrcData(&stNnieCfg, &s_stSegnetNnieParam, &stInputDataIdx);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, SEGNET_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_FillSrcData failed!\n");

    /* NNIE process(process the 0-th segment) */
    if (g_stop_signal == HI_TRUE) {
        SAMPLE_SVP_NNIE_Segnet_Stop();
        return;
    }
    stProcSegIdx.u32SegIdx = 0;
    /*推理过程，结果在s_stSegnetNnieParam中*/
    s32Ret = SAMPLE_SVP_NNIE_Forward(&s_stSegnetNnieParam, &stInputDataIdx, &stProcSegIdx, HI_TRUE);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, SEGNET_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Forward failed!\n");

    /* print report result */
    s32Ret = SAMPLE_SVP_NNIE_PrintReportResult(&s_stSegnetNnieParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, SEGNET_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_PrintReportResult failed!\n");

    SAMPLE_SVP_TRACE_INFO("Segnet is successfully processed!\n");

SEGNET_FAIL_0:
    SAMPLE_SVP_NNIE_Cnn_Deinit(&s_stSegnetNnieParam, NULL, &s_stSegnetModel);
    SAMPLE_COMM_SVP_CheckSysExit();
}
