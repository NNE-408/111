import os
import platform
import ctypes
import time
import struct
from HCNetSDK import *

# 系统环境标识
WINDOWS_FLAG = True


# 定义 NET_DVR_JPEGPARA 结构体（保持不变）
class NET_DVR_JPEGPARA(ctypes.Structure):
    _fields_ = [
        ("wPicSize", ctypes.c_ushort),  # 图片分辨率
        ("wPicQuality", ctypes.c_ushort),  # 图片质量
    ]


# 新增结构体定义（严格按Windows文档校准）
class NET_VCR_RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long)
    ]


class NET_DVR_JPEGPICTURE_WITH_APPENDDATA(ctypes.Structure):
    _fields_ = [
        ("dwSize", c_int32),  # 结构体大小
        ("dwChannel", c_int32),  # 通道号
        ("dwJpegPicLen", c_int32),  # JPEG图片长度
        ("pJpegPicBuf", ctypes.POINTER(ctypes.c_ubyte)),  # JPEG图片缓冲区指针
        ("dwJpegPicWidth", c_int32),  # 图像宽度
        ("dwJpegPicHeight", c_int32),  # 图像高度
        ("dwP2PDatalen", c_int32),  # 全屏测温数据长度
        ("pP2PDataBuff", ctypes.POINTER(ctypes.c_ubyte)),  # 测温数据指针
        ("byISFreezedata", ctypes.c_ubyte),  # 是否冻结数据
        ("byRes1", ctypes.c_ubyte * 3),  # 保留字段
        ("dwVisiblePicLen", c_int32),  # 可见光图片长度
        ("pVisiblePicBuf", ctypes.POINTER(ctypes.c_ubyte)),  # 可见光图片指针
        ("byRes2", ctypes.c_ubyte * 8),  # 保留字段（64位对齐）
        ("struThermalValidRect", NET_VCR_RECT),  # 热成像有效区域
        ("struVisibleValidRect", NET_VCR_RECT),  # 可见光有效区域
        ("byRes", ctypes.c_ubyte * 208)  # 保留字段
    ]


# 保留您原有的函数定义
def GetPlatform():
    sysstr = platform.system()
    print('' + sysstr)
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


if __name__ == '__main__':
    try:
        print("\n=== 调用全屏测温接口前参数检查 ===")
        print(f"通道号: 2 (固定值)")
        print(f"结构体大小: {NET_DVR_JPEGPICTURE_WITH_APPENDDATA.dwSize} (应为{ctypes.sizeof(NET_DVR_JPEGPICTURE_WITH_APPENDDATA)})")
        print(f"JPEG缓冲区地址: {hex(ctypes.addressof(NET_DVR_JPEGPICTURE_WITH_APPENDDATA.pJpegPicBuf)) if NET_DVR_JPEGPICTURE_WITH_APPENDDATA.pJpegPicBuf else 'NULL'}")
        print(f"温度缓冲区地址: {hex(ctypes.addressof(NET_DVR_JPEGPICTURE_WITH_APPENDDATA.temp_buf)) if NET_DVR_JPEGPICTURE_WITH_APPENDDATA.temp_buf else 'NULL'}")
        print(f"结构体地址: {hex(ctypes.addressof(NET_DVR_JPEGPICTURE_WITH_APPENDDATA.thermal_data))}")

        ret = NET_DVR_CaptureJPEGPicture_WithAppendData(NET_DVR_JPEGPICTURE_WITH_APPENDDATA.UserID, 2, byref(NET_DVR_JPEGPICTURE_WITH_APPENDDATA.thermal_data))
        if not ret:
            error_code = sdk.NET_DVR_GetLastError()
            print(f"错误码: {error_code}")
            error_messages = {
                2: "参数错误 - 请检查结构体定义和缓冲区大小",
                7: "缓冲区不足 - 尝试增大MAX_JPEG_SIZE/MAX_TEMP_SIZE",
                10: "SDK未初始化 - 检查初始化流程",
                6: "设备不支持 - 确认设备型号支持全屏测温"
            }
            print(f"错误解释: {error_messages.get(error_code, '未知错误')}")
        else:
            print(f"JPEG数据长度: {thermal_data.dwJpegPicLen}")
            print(f"温度数据长度: {thermal_data.dwP2PDatalen}")
            if thermal_data.dwP2PDatalen >= 4:
                first_temp = struct.unpack('<f', bytes(temp_buf[:4]))[0]
                print(f"首个温度值: {first_temp:.2f}°C")
    except Exception as e:
        print(f"测温接口调用异常: {str(e)}")
        # 保留您原有的初始化流程
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

        # 用户登录（保持不变）
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

            # 保留原有的普通抓图功能
            jpeg_para = NET_DVR_JPEGPARA()
            jpeg_para.wPicSize = 2
            jpeg_para.wPicQuality = 100

            image_count = 0
            try:
                while True:
                    image_count += 1
                    save_path = f"./capture_{image_count}.jpg"

                    # 1. 普通抓图（保持原有功能）
                    if sdk.NET_DVR_CaptureJPEGPicture(UserID, 2, byref(jpeg_para), save_path.encode('utf-8')):
                        print(f"抓图成功，保存路径：{save_path}")

                        # 2. 新增全屏测温抓图（独立调用）
                        thermal_data = NET_DVR_JPEGPICTURE_WITH_APPENDDATA()
                        thermal_data.dwSize = ctypes.sizeof(NET_DVR_JPEGPICTURE_WITH_APPENDDATA)

                        # 分配缓冲区（Linux下需要手动分配）
                        MAX_JPEG_SIZE = 8 * 1024 * 1024  # 8MB
                        MAX_TEMP_SIZE = 2 * 1024 * 1024  # 2MB
                        jpeg_buf = (ctypes.c_ubyte * MAX_JPEG_SIZE)()
                        temp_buf = (ctypes.c_ubyte * MAX_TEMP_SIZE)()

                        thermal_data.pJpegPicBuf = ctypes.cast(jpeg_buf, ctypes.POINTER(ctypes.c_ubyte))
                        thermal_data.pP2PDataBuff = ctypes.cast(temp_buf, ctypes.POINTER(ctypes.c_ubyte))

                        # 调用全屏测温接口（注意Linux下可能接口名不同）
                        if hasattr(sdk, 'NET_DVR_CaptureJPEGPicture_WithAppendData'):
                            if sdk.NET_DVR_CaptureJPEGPicture_WithAppendData(UserID, 2, byref(thermal_data)):
                                if thermal_data.dwP2PDatalen > 0:
                                    # 解析温度数据（示例：获取第一个温度值）
                                    first_temp = struct.unpack('<f', bytes(temp_buf[:4]))[0]
                                    print(f"获取到温度数据，长度: {thermal_data.dwP2PDatalen}字节")
                                    print(f"首个温度值: {first_temp:.2f}°C")
                                else:
                                    print("未获取到温度数据")
                            else:
                                print(f"全屏测温失败，错误码: {sdk.NET_DVR_GetLastError()}")
                        else:
                            print("当前SDK不支持全屏测温功能")

                    time.sleep(3)

            except KeyboardInterrupt:
                print("用户中断程序...")

            finally:
                if UserID >= 0:
                    sdk.NET_DVR_Logout(UserID)
                sdk.NET_DVR_Cleanup()
