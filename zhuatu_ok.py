import os
import platform
import ctypes
import time
from HCNetSDK import *

# 系统环境标识
WINDOWS_FLAG = platform.system() == "Windows"


# 定义NET_VCA_RECT结构体（用于有效区域坐标）
class NET_VCA_RECT(ctypes.Structure):
    _fields_ = [
        ("fX", ctypes.c_float),  # 水平坐标(0.001~1)
        ("fY", ctypes.c_float),  # 垂直坐标(0.001~1)
        ("fWidth", ctypes.c_float),  # 宽度(0.001~1)
        ("fHeight", ctypes.c_float)  # 高度(0.001~1)
    ]


# 定义全屏测温抓图结构体（严格对齐官方文档）
class NET_DVR_JPEGPICTURE_WITH_APPENDDATA(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_ulong),  # 结构体大小（必须设置）
        ("dwChannel", ctypes.c_ulong),  # 通道号
        ("dwJpegPicLen", ctypes.c_ulong),  # Jpeg图片长度
        ("pJpegPicBuf", ctypes.POINTER(ctypes.c_ubyte)),  # Jpeg图片指针
        ("dwJpegPicWidth", ctypes.c_ulong),  # 图像宽度
        ("dwJpegPicHeight", ctypes.c_ulong),  # 图像高度
        ("dwFZDataLen", ctypes.c_ulong),  # 全屏测温数据长度
        ("pFZDataBuf", ctypes.POINTER(ctypes.c_ubyte)),  # 全屏测温数据指针
        ("byThermalPic", ctypes.c_ubyte),  # 是否包含热成像数据：0-否，1-是
        ("byRes1", ctypes.c_ubyte * 3),  # 保留字段
        ("dwViziblePicLen", ctypes.c_ulong),  # 可见光图片长度（可选）
        ("pViziblePicBuf", ctypes.POINTER(ctypes.c_ubyte)),  # 可见光图片指针（可选）
        ("byRes2", ctypes.c_ubyte * 8),  # 64位系统对齐填充
        ("struThermalValidRect", NET_VCA_RECT),  # 热成像有效区域
        ("struVizibleValidRect", NET_VCA_RECT),  # 可见光有效区域
        ("byRes", ctypes.c_ubyte * 20)  # 保留字段
    ]


def init_sdk():
    """初始化SDK环境（整合您的库加载逻辑）"""
    # 获取系统平台
    if WINDOWS_FLAG:
        os.chdir(r'./lib/win')
        sdk = ctypes.CDLL(r'./HCNetSDK.dll')
    else:
        os.chdir(r'./lib/linux')
        sdk = ctypes.CDLL(r'./libhcnetsdk.so')

    # 设置组件库和SSL库加载路径（您的原始逻辑）
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

    # 初始化SDK
    if not sdk.NET_DVR_Init():
        raise RuntimeError(f"SDK初始化失败，错误码: {sdk.NET_DVR_GetLastError()}")

    # 启用日志（您的原始逻辑）
    sdk.NET_DVR_SetLogToFile(3, bytes(r'./sdklog', encoding="utf-8"), False)

    # 通用参数配置（您的原始逻辑）
    sdkCfg = NET_DVR_LOCAL_GENERAL_CFG()
    sdkCfg.byAlarmJsonPictureSeparate = 1
    sdk.NET_DVR_SetSDKLocalCfg(17, byref(sdkCfg))

    return sdk


def login_device(sdk):
    """登录设备（完全使用您的登录逻辑）"""
    struLoginInfo = NET_DVR_USER_LOGIN_INFO()
    struLoginInfo.bUseAsynLogin = 0  # 同步登录方式
    struLoginInfo.sDeviceAddress = bytes("192.168.2.64", "ascii")  # 设备IP地址
    struLoginInfo.wPort = 8000  # 设备服务端口
    struLoginInfo.sUserName = bytes("admin", "ascii")  # 设备登录用户名
    struLoginInfo.sPassword = bytes("wwf123456", "ascii")  # 设备登录密码
    struLoginInfo.byLoginMode = 0

    struDeviceInfoV40 = NET_DVR_DEVICEINFO_V40()
    UserID = sdk.NET_DVR_Login_V40(byref(struLoginInfo), byref(struDeviceInfoV40))

    if UserID < 0:
        raise RuntimeError(f"登录失败，错误码: {sdk.NET_DVR_GetLastError()}")

    print('登录成功，设备序列号：%s' % str(struDeviceInfoV40.struDeviceV30.sSerialNumber, encoding="utf8"))
    return UserID


def capture_with_thermal(sdk, user_id, channel=1):
    """执行全屏测温抓图"""
    # 分配缓冲区
    jpeg_buf = (ctypes.c_ubyte * 4 * 1024 * 1024)()  # 4MB JPEG缓冲区
    thermal_buf = (ctypes.c_ubyte * 2 * 1024 * 1024)()  # 2MB测温数据缓冲区

    # 初始化结构体
    jpeg_para = NET_DVR_JPEGPICTURE_WITH_APPENDDATA()
    jpeg_para.dwSize = ctypes.sizeof(NET_DVR_JPEGPICTURE_WITH_APPENDDATA)
    jpeg_para.dwChannel = channel
    jpeg_para.byThermalPic = 1  # 启用测温数据

    # 设置缓冲区指针
    jpeg_para.pJpegPicBuf = ctypes.cast(jpeg_buf, ctypes.POINTER(ctypes.c_ubyte))
    jpeg_para.pFZDataBuf = ctypes.cast(thermal_buf, ctypes.POINTER(ctypes.c_ubyte))

    # 调用抓图接口
    if not sdk.NET_DVR_CaptureJPEGPicture_WithAppendData(user_id, channel, byref(jpeg_para)):
        raise RuntimeError(f"抓图失败，错误码: {sdk.NET_DVR_GetLastError()}")

    # 保存JPEG图片
    if jpeg_para.dwJpegPicLen > 0:
        with open(f'thermal_capture_{int(time.time())}.jpg', 'wb') as f:
            f.write(bytes(jpeg_buf[:jpeg_para.dwJpegPicLen]))
        print(f"图片已保存，大小: {jpeg_para.dwJpegPicLen}字节")

    # 处理测温数据（需根据设备协议解析）
    if jpeg_para.dwFZDataLen > 0:
        print(f"获取到测温数据，长度: {jpeg_para.dwFZDataLen}字节")
        # 此处添加测温数据解析代码...


if __name__ == '__main__':
    try:
        # 初始化SDK（整合您的初始化逻辑）
        sdk = init_sdk()

        # 登录设备（完全使用您的登录逻辑）
        user_id = login_device(sdk)

        # 执行全屏测温抓图
        while True:
            capture_with_thermal(sdk, user_id)
            time.sleep(3)  # 每隔3秒抓一次图

    except KeyboardInterrupt:
        print("\n用户中断程序，准备退出...")
    except Exception as e:
        print(f"[ERROR] 发生异常: {str(e)}")
    finally:
        # 清理资源（保持您的清理逻辑）
        if 'user_id' in locals() and user_id >= 0:
            flag = sdk.NET_DVR_Logout(user_id)
            print("设备注销成功" if flag else f"设备注销失败，错误码: {sdk.NET_DVR_GetLastError()}")
        if 'sdk' in locals():
            sdk.NET_DVR_Cleanup()
        print("程序退出")