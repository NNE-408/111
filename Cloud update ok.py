import os
import platform
import ctypes
import time
import struct
import threading
from HCNetSDK import *

# ========================= 结构体定义 =========================

WINDOWS_FLAG = True

class NET_DVR_JPEGPARA(ctypes.Structure):
    _fields_ = [
        ("wPicSize", ctypes.c_ushort),
        ("wPicQuality", ctypes.c_ushort),
    ]

class NET_VCA_RECT(ctypes.Structure):
    _fields_ = [
        ("fX", ctypes.c_float),
        ("fY", ctypes.c_float),
        ("fWidth", ctypes.c_float),
        ("fHeight", ctypes.c_float)
    ]

class NET_DVR_JPEGPICTURE_WITH_APPENDDATA(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_int32),
        ("dwChannel", ctypes.c_int32),
        ("dwJpegPicLen", ctypes.c_int32),
        ("pJpegPicBuf", ctypes.POINTER(ctypes.c_ubyte)),
        ("dwJpegPicWidth", ctypes.c_int32),
        ("dwJpegPicHeight", ctypes.c_int32),
        ("dwP2PDatalen", ctypes.c_int32),
        ("pP2PDataBuff", ctypes.POINTER(ctypes.c_ubyte)),
        ("byISFreezedata", ctypes.c_ubyte),
        ("byRes1", ctypes.c_ubyte * 3),
        ("dwVisiblePicLen", ctypes.c_int32),
        ("pVisiblePicBuf", ctypes.POINTER(ctypes.c_ubyte)),
        ("byRes2", ctypes.c_ubyte * 8),
        ("struThermalValidRect", NET_VCA_RECT),
        ("struVisibleValidRect", NET_VCA_RECT),
        ("byRes", ctypes.c_ubyte * 208)
    ]

class NET_DVR_PREVIEWINFO(ctypes.Structure):
    _fields_ = [
        ("lChannel", ctypes.c_long),
        ("dwStreamType", ctypes.c_ulong),
        ("dwLinkMode", ctypes.c_ulong),
        ("hPlayWnd", ctypes.c_void_p),
        ("bBlocked", ctypes.c_bool),
        ("bPassbackRecord", ctypes.c_bool),
        ("byPreviewMode", ctypes.c_ubyte),
        ("byStreamID", ctypes.c_ubyte * 32),
        ("byProtoType", ctypes.c_ubyte),
        ("byRes1", ctypes.c_ubyte * 2),
        ("byVideoCodingType", ctypes.c_ubyte),
        ("dwDisplayBufNum", ctypes.c_ulong),
        ("byRes", ctypes.c_ubyte * 216)
    ]

# ========================= 回调函数定义 =========================

REALDATACALLBACK = ctypes.CFUNCTYPE(
    None, ctypes.c_long, ctypes.c_ulong,
    ctypes.POINTER(ctypes.c_ubyte), ctypes.c_ulong, ctypes.c_ulong
)

@REALDATACALLBACK
def RealDataCallBack(lRealHandle, dwDataType, pBuffer, dwBufSize, dwUser):
    if dwDataType == 2:
        with open("stream_output.ps", "ab") as f:
            f.write(ctypes.string_at(pBuffer, dwBufSize))
        print(f"[码流] 接收到 {dwBufSize} 字节")

# ========================= 流文件增长测试 =========================

def monitor_stream_file_growth():
    print("开始监控 stream_output.ps 文件是否持续增长...")
    previous_size = 0
    while True:
        try:
            current_size = os.path.getsize("stream_output.ps")
            if current_size > previous_size:
                print(f"[增长中] 文件大小: {current_size} 字节")
            else:
                print(f"[无变化] 文件大小: {current_size} 字节")
            previous_size = current_size
            time.sleep(1)
        except FileNotFoundError:
            print("[等待中] 文件尚未创建...")
            time.sleep(1)

# ========================= 实时码流启动函数 =========================

def start_realplay_and_stream(sdk, UserID):
    previewInfo = NET_DVR_PREVIEWINFO()
    previewInfo.lChannel = 1
    previewInfo.dwStreamType = 0
    previewInfo.dwLinkMode = 0
    previewInfo.hPlayWnd = 0
    previewInfo.bBlocked = 1

    sdk.NET_DVR_RealPlay_V40.argtypes = [ctypes.c_long, ctypes.POINTER(NET_DVR_PREVIEWINFO), REALDATACALLBACK, ctypes.c_void_p]
    sdk.NET_DVR_RealPlay_V40.restype = ctypes.c_long

    realHandle = sdk.NET_DVR_RealPlay_V40(UserID, ctypes.byref(previewInfo), RealDataCallBack, None)
    if realHandle < 0:
        print("启动实时预览失败，错误码：", sdk.NET_DVR_GetLastError())
    else:
        print("实时码流通道建立成功，保存路径：stream_output.ps")
        threading.Thread(target=monitor_stream_file_growth, daemon=True).start()

# ========================= SDK初始化辅助 =========================

def GetPlatform():
    sysstr = platform.system()
    if sysstr != "Windows":
        global WINDOWS_FLAG
        WINDOWS_FLAG = False

def SetSDKInitCfg():
    if WINDOWS_FLAG:
        strPath = os.getcwd().encode('gbk')
        sdk_ComPath = NET_DVR_LOCAL_SDK_PATH()
        sdk_ComPath.sPath = strPath
        sdk.NET_DVR_SetSDKInitCfg(2, byref(sdk_ComPath))
        sdk.NET_DVR_SetSDKInitCfg(3, create_string_buffer(strPath + b'\libcrypto-1_1-x64.dll'))
        sdk.NET_DVR_SetSDKInitCfg(4, create_string_buffer(strPath + b'\libssl-1_1-x64.dll'))
    else:
        strPath = os.getcwd().encode('utf-8')
        sdk_ComPath = NET_DVR_LOCAL_SDK_PATH()
        sdk_ComPath.sPath = strPath
        sdk.NET_DVR_SetSDKInitCfg(2, byref(sdk_ComPath))
        sdk.NET_DVR_SetSDKInitCfg(3, create_string_buffer(strPath + b'/libcrypto.so.1.1'))
        sdk.NET_DVR_SetSDKInitCfg(4, create_string_buffer(strPath + b'/libssl.so.1.1'))

def NET_DVR_CaptureJPEGPicture_WithAppendData(lUserID, lChannel, lpJpegWithAppend):
    func = sdk.NET_DVR_CaptureJPEGPicture_WithAppendData
    func.argtypes = [ctypes.c_long, ctypes.c_long, ctypes.POINTER(NET_DVR_JPEGPICTURE_WITH_APPENDDATA)]
    func.restype = ctypes.c_bool
    return func(lUserID, lChannel, lpJpegWithAppend)

# ========================= 主程序入口 =========================

if __name__ == '__main__':
    try:
        GetPlatform()
        if WINDOWS_FLAG:
            os.chdir(r'./lib/win')
            sdk = ctypes.CDLL(r'./HCNetSDK.dll')
        else:
            os.chdir(r'./lib/linux')
            sdk = ctypes.CDLL(r'./libhcnetsdk.so')

        SetSDKInitCfg()
        sdk.NET_DVR_Init()
        sdk.NET_DVR_SetLogToFile(3, bytes(r'./sdklog', encoding="utf-8"), False)

        struLoginInfo = NET_DVR_USER_LOGIN_INFO()
        struLoginInfo.bUseAsynLogin = 0
        struLoginInfo.sDeviceAddress = bytes("192.168.2.64", "ascii")
        struLoginInfo.wPort = 8000
        struLoginInfo.sUserName = bytes("admin", "ascii")
        struLoginInfo.sPassword = bytes("wwf123456", "ascii")
        struDeviceInfoV40 = NET_DVR_DEVICEINFO_V40()

        UserID = sdk.NET_DVR_Login_V40(byref(struLoginInfo), byref(struDeviceInfoV40))
        if UserID < 0:
            print("Login failed, error code: %d" % sdk.NET_DVR_GetLastError())
            sdk.NET_DVR_Cleanup()
        else:
            print('登录成功，设备序列号：%s' % str(struDeviceInfoV40.struDeviceV30.sSerialNumber, encoding="utf8"))
            start_realplay_and_stream(sdk, UserID)

            # 保持主线程阻塞运行
            while True:
                time.sleep(1)

    except Exception as e:
        print(f"测温接口调用异常: {str(e)}")
