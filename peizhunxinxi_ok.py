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
class NET_VCA_RECT(ctypes.Structure):
    """
    标准区域结构体（与SDK文档严格一致）
    """
    _fields_ = [
        ("fX", ctypes.c_float),  # 左上角X坐标 [0.0,1.0]
        ("fY", ctypes.c_float),  # 左上角Y坐标 [0.0,1.0]
        ("fWidth", ctypes.c_float),  # 宽度 [0.0,1.0]
        ("fHeight", ctypes.c_float)  # 高度 [0.0,1.0]
    ]


class NET_VCA_RECT(ctypes.Structure):
    """
    区域参数结构体（严格匹配SDK手册定义）
    坐标范围: 0.0~1.0 表示相对位置/尺寸
    """
    _fields_ = [
        ("fX", ctypes.c_float),  # 左上角X坐标 [0.000,1.000]
        ("fY", ctypes.c_float),  # 左上角Y坐标 [0.000,1.000]
        ("fWidth", ctypes.c_float),  # 宽度 [0.000,1.000]
        ("fHeight", ctypes.c_float)  # 高度 [0.000,1.000]
    ]


class NET_DVR_JPEGPICTURE_WITH_APPENDDATA(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_int32),  # 结构体大小
        ("dwChannel", ctypes.c_int32),  # 通道号
        ("dwJpegPicLen", ctypes.c_int32),  # JPEG图片长度
        ("pJpegPicBuf", ctypes.POINTER(ctypes.c_ubyte)),  # JPEG图片缓冲区指针
        ("dwJpegPicWidth", ctypes.c_int32),  # 图像宽度
        ("dwJpegPicHeight", ctypes.c_int32),  # 图像高度
        ("dwP2PDatalen", ctypes.c_int32),  # 全屏测温数据长度
        ("pP2PDataBuff", ctypes.POINTER(ctypes.c_ubyte)),  # 测温数据指针
        ("byISFreezedata", ctypes.c_ubyte),  # 是否冻结数据
        ("byRes1", ctypes.c_ubyte * 3),  # 保留字段
        ("dwVisiblePicLen", ctypes.c_int32),  # 可见光图片长度
        ("pVisiblePicBuf", ctypes.POINTER(ctypes.c_ubyte)),  # 可见光图片指针
        ("byRes2", ctypes.c_ubyte * 8),  # 保留字段（64位对齐）
        ("struThermalValidRect", NET_VCA_RECT),  # 热成像有效区域
        ("struVisibleValidRect", NET_VCA_RECT),  # 可见光有效区域
        ("byRes", ctypes.c_ubyte * 208)  # 保留字段
    ]


# 保留原有的函数定义
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

            # 分配缓冲区（Linux下需要手动分配）
            MAX_JPEG_SIZE = 8 * 1024 * 1024  # 8MB
            MAX_TEMP_SIZE = 2 * 1024 * 1024  # 2MB
            jpeg_buf = (ctypes.c_ubyte * MAX_JPEG_SIZE)()
            temp_buf = (ctypes.c_ubyte * MAX_TEMP_SIZE)()

            thermal_data = NET_DVR_JPEGPICTURE_WITH_APPENDDATA()
            thermal_data.dwSize = ctypes.sizeof(NET_DVR_JPEGPICTURE_WITH_APPENDDATA)
            thermal_data.pJpegPicBuf = ctypes.cast(jpeg_buf, ctypes.POINTER(ctypes.c_ubyte))
            thermal_data.pP2PDataBuff = ctypes.cast(temp_buf, ctypes.POINTER(ctypes.c_ubyte))

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
                                    print("\n=== NET_DVR_JPEGPICTURE_WITH_APPENDDATA 结构体信息 ===")

                                    # 基础字段
                                    print(f"[dwSize] 结构体大小: {thermal_data.dwSize} bytes")
                                    print(f"[dwChannel] 通道号: {thermal_data.dwChannel}")
                                    print(f"[dwJpegPicLen] JPEG长度: {thermal_data.dwJpegPicLen} bytes")
                                    print(
                                        f"[pJpegPicBuf] JPEG缓冲区地址: {hex(ctypes.addressof(thermal_data.pJpegPicBuf.contents)) if thermal_data.dwJpegPicLen > 0 else 'NULL'}")
                                    print(f"[dwJpegPicWidth] 图像宽度: {thermal_data.dwJpegPicWidth} pixels")
                                    print(f"[dwJpegPicHeight] 图像高度: {thermal_data.dwJpegPicHeight} pixels")

                                    # 测温数据字段
                                    print(f"\n[测温数据]")
                                    print(f"[dwP2PDatalen] 数据长度: {thermal_data.dwP2PDatalen} bytes")
                                    print(
                                        f"[pP2PDataBuff] 温度数据地址: {hex(ctypes.addressof(thermal_data.pP2PDataBuff.contents)) if thermal_data.dwP2PDatalen > 0 else 'NULL'}")
                                    print(f"[byISFreezedata] 是否冻结: {'是' if thermal_data.byISFreezedata else '否'}")

                                    # 可见光数据字段
                                    # print(f"\n[可见光数据]")
                                    # print(f"[dwVisiblePicLen] 可见光长度: {thermal_data.dwVisiblePicLen} bytes")
                                    # print(
                                    #     f"[pVisiblePicBuf] 可见光地址: {hex(ctypes.addressof(thermal_data.pVisiblePicBuf.contents)) if thermal_data.dwVisiblePicLen > 0 else 'NULL'}")

                                    # 有效区域
                                    print(f"\n[热成像有效区域]")
                                    print(f" {thermal_data.struThermalValidRect.fX}")
                                    print(f" {thermal_data.struThermalValidRect.fY}")
                                    print(f" {thermal_data.struVisibleValidRect.fHeight}")
                                    print(f" {thermal_data.struVisibleValidRect.fWidth}")

                                    print(f"\n[可见光有效区域]")
                                    print(f" {thermal_data.struVisibleValidRect.fX}")
                                    print(f" {thermal_data.struVisibleValidRect.fY}")
                                    print(f" {thermal_data.struThermalValidRect.fHeight}")
                                    print(f" {thermal_data.struThermalValidRect.fWidth}")

                                    # 保留字段（仅显示状态）
                                    print("\n[保留字段状态]")
                                    print(f"byRes1: {bytes(thermal_data.byRes1).hex()}")
                                    print(f"byRes2: {bytes(thermal_data.byRes2).hex()}")
                                    print(f"byRes: {bytes(thermal_data.byRes)[:16].hex()}... (前16字节)")
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
    except Exception as e:
        print(f"测温接口调用异常: {str(e)}")
