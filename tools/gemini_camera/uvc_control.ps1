param(
    [string]$Match = "vid_2bc5&pid_0511",
    [ValidateSet("list", "brighten", "set", "auto")]
    [string]$Action = "list",
    [int]$Brightness = 80,
    [int]$Gain = 64,
    [int]$Exposure = -4
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$TmpDir = Join-Path $ScriptDir "tmp"
New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null
$env:TEMP = $TmpDir
$env:TMP = $TmpDir

$source = @"
using System;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;

namespace UvcCtl {
    [ComImport, Guid("62BE5D10-60EB-11d0-BD3B-00A0C911CE86")]
    public class CreateDevEnum {}

    [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("29840822-5B84-11D0-BD3B-00A0C911CE86")]
    public interface ICreateDevEnum {
        int CreateClassEnumerator([In] ref Guid pType, out IEnumMoniker ppEnumMoniker, int dwFlags);
    }

    [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("55272A00-42CB-11CE-8135-00AA004BB851")]
    public interface IPropertyBag {
        int Read([MarshalAs(UnmanagedType.LPWStr)] string pszPropName, [In, Out] ref object pVar, IntPtr pErrorLog);
        int Write([MarshalAs(UnmanagedType.LPWStr)] string pszPropName, ref object pVar);
    }

    [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("C6E13360-30AC-11d0-A18C-00A0C9118956")]
    public interface IAMVideoProcAmp {
        int GetRange(int Property, out int pMin, out int pMax, out int pSteppingDelta, out int pDefault, out int pCapsFlags);
        int Set(int Property, int lValue, int Flags);
        int Get(int Property, out int lValue, out int Flags);
    }

    [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("C6E13370-30AC-11d0-A18C-00A0C9118956")]
    public interface IAMCameraControl {
        int GetRange(int Property, out int pMin, out int pMax, out int pSteppingDelta, out int pDefault, out int pCapsFlags);
        int Set(int Property, int lValue, int Flags);
        int Get(int Property, out int lValue, out int Flags);
    }

    public static class Program {
        static readonly Guid VideoInputDeviceCategory = new Guid("860BB310-5D01-11d0-BD3B-00A0C911CE86");
        static readonly Guid PropertyBagGuid = new Guid("55272A00-42CB-11CE-8135-00AA004BB851");
        static readonly Guid BaseFilterGuid = new Guid("56a86895-0ad4-11ce-b03a-0020af0ba770");

        const int Manual = 0x2;
        const int Auto = 0x1;
        const int Brightness = 0;
        const int Contrast = 1;
        const int Saturation = 3;
        const int Gain = 9;
        const int Exposure = 4;

        public static int Main(string[] args) {
            string match = args.Length > 0 ? args[0].ToLowerInvariant() : "vid_2bc5&pid_0511";
            string action = args.Length > 1 ? args[1].ToLowerInvariant() : "list";
            int brightness = args.Length > 2 ? int.Parse(args[2]) : 80;
            int gain = args.Length > 3 ? int.Parse(args[3]) : 64;
            int exposure = args.Length > 4 ? int.Parse(args[4]) : -4;

            var devEnum = (ICreateDevEnum)new CreateDevEnum();
            Guid videoInputDeviceCategory = VideoInputDeviceCategory;
            IEnumMoniker enumMoniker;
            int hr = devEnum.CreateClassEnumerator(ref videoInputDeviceCategory, out enumMoniker, 0);
            if (hr != 0 || enumMoniker == null) {
                Console.WriteLine("No DirectShow video devices found.");
                return 1;
            }

            IMoniker[] monikers = new IMoniker[1];
            IntPtr fetched = IntPtr.Zero;
            bool matched = false;
            while (enumMoniker.Next(1, monikers, fetched) == 0) {
                string name = ReadBag(monikers[0], "FriendlyName");
                string path = ReadBag(monikers[0], "DevicePath");
                Console.WriteLine("Device: " + name);
                Console.WriteLine("  Path: " + path);

                if ((path ?? "").ToLowerInvariant().Contains(match) || (name ?? "").ToLowerInvariant().Contains(match)) {
                    matched = true;
                    object filterObj;
                    Guid baseFilterGuid = BaseFilterGuid;
                    monikers[0].BindToObject(null, null, ref baseFilterGuid, out filterObj);
                    var vpa = filterObj as IAMVideoProcAmp;
                    var cam = filterObj as IAMCameraControl;

                    if (vpa == null) Console.WriteLine("  IAMVideoProcAmp: unavailable");
                    else {
                        DumpVpa(vpa, Brightness, "Brightness");
                        DumpVpa(vpa, Contrast, "Contrast");
                        DumpVpa(vpa, Saturation, "Saturation");
                        DumpVpa(vpa, Gain, "Gain");
                    }

                    if (cam == null) Console.WriteLine("  IAMCameraControl: unavailable");
                    else DumpCam(cam, Exposure, "Exposure");

                    if (action == "brighten" || action == "set") {
                        Console.WriteLine("Applying manual brightening settings...");
                        TrySetCam(cam, Exposure, exposure, Manual, "Exposure");
                        TrySetVpa(vpa, Gain, gain, Manual, "Gain");
                        TrySetVpa(vpa, Brightness, brightness, Manual, "Brightness");
                        Console.WriteLine("After:");
                        if (vpa != null) {
                            DumpVpa(vpa, Brightness, "Brightness");
                            DumpVpa(vpa, Gain, "Gain");
                        }
                        if (cam != null) DumpCam(cam, Exposure, "Exposure");
                    }
                    if (action == "auto") {
                        Console.WriteLine("Restoring conservative automatic exposure settings...");
                        TrySetCam(cam, Exposure, exposure, Auto, "ExposureAuto");
                        TrySetVpa(vpa, Gain, 0, Manual, "Gain");
                        TrySetVpa(vpa, Brightness, 0, Manual, "Brightness");
                        Console.WriteLine("After:");
                        if (vpa != null) {
                            DumpVpa(vpa, Brightness, "Brightness");
                            DumpVpa(vpa, Gain, "Gain");
                        }
                        if (cam != null) DumpCam(cam, Exposure, "Exposure");
                    }
                    Marshal.ReleaseComObject(filterObj);
                }
                Marshal.ReleaseComObject(monikers[0]);
            }

            if (!matched) {
                Console.WriteLine("No matched device for: " + match);
                return 2;
            }
            return 0;
        }

        static string ReadBag(IMoniker moniker, string key) {
            object bagObj;
            Guid propertyBagGuid = PropertyBagGuid;
            moniker.BindToStorage(null, null, ref propertyBagGuid, out bagObj);
            var bag = (IPropertyBag)bagObj;
            object value = "";
            try { bag.Read(key, ref value, IntPtr.Zero); }
            catch { value = ""; }
            Marshal.ReleaseComObject(bagObj);
            return value == null ? "" : value.ToString();
        }

        static void DumpVpa(IAMVideoProcAmp vpa, int prop, string name) {
            int min, max, step, def, caps, val, flags;
            int hr = vpa.GetRange(prop, out min, out max, out step, out def, out caps);
            if (hr != 0) return;
            vpa.Get(prop, out val, out flags);
            Console.WriteLine(String.Format("  {0}: value={1} flags={2} range=[{3},{4}] step={5} default={6} caps={7}", name, val, flags, min, max, step, def, caps));
        }

        static void DumpCam(IAMCameraControl cam, int prop, string name) {
            int min, max, step, def, caps, val, flags;
            int hr = cam.GetRange(prop, out min, out max, out step, out def, out caps);
            if (hr != 0) return;
            cam.Get(prop, out val, out flags);
            Console.WriteLine(String.Format("  {0}: value={1} flags={2} range=[{3},{4}] step={5} default={6} caps={7}", name, val, flags, min, max, step, def, caps));
        }

        static void TrySetVpa(IAMVideoProcAmp vpa, int prop, int value, int flags, string name) {
            if (vpa == null) return;
            int min, max, step, def, caps;
            if (vpa.GetRange(prop, out min, out max, out step, out def, out caps) != 0) return;
            int clamped = Math.Max(min, Math.Min(max, value));
            int hr = vpa.Set(prop, clamped, flags);
            Console.WriteLine(String.Format("  set {0}={1} hr=0x{2:X8}", name, clamped, hr));
        }

        static void TrySetCam(IAMCameraControl cam, int prop, int value, int flags, string name) {
            if (cam == null) return;
            int min, max, step, def, caps;
            if (cam.GetRange(prop, out min, out max, out step, out def, out caps) != 0) return;
            int clamped = Math.Max(min, Math.Min(max, value));
            int hr = cam.Set(prop, clamped, flags);
            Console.WriteLine(String.Format("  set {0}={1} hr=0x{2:X8}", name, clamped, hr));
        }
    }
}
"@

Add-Type -TypeDefinition $source -Language CSharp -OutputAssembly (Join-Path $TmpDir "UvcCtl.exe") -OutputType ConsoleApplication
& (Join-Path $TmpDir "UvcCtl.exe") $Match $Action $Brightness $Gain $Exposure
