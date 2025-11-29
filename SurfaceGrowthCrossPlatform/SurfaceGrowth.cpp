//---------------------------------------------------------------------
// SurfaceGrowth.cpp - program's entry point, input/output management.
// (c) 2010 - 2025 Mykola Prodanov
//---------------------------------------------------------------------

#include "SurfaceGrowthProto.h" // Contains necessary definitions and function prototypes.

///////////
// Globals.
///////////
cudaDeviceProp g_hDeviceProp;                   // Device properties.

char   g_szAppName[] = TEXT("SurfaceGrowth");  // Application name.
char*  g_szCmdLine;                            // Command line for CudaInitW.

// File variables.
// Beginning of the file names.
char   g_szResult[MAX_PATH] = TEXT(""),
        g_szPdb[MAX_PATH] = TEXT("sugr"),
        g_szRdf[MAX_PATH] = TEXT("rdf");
FILE*   g_fResult = NULL;                       // Output file descriptor.
FILE*   g_fPdb = NULL;                          // Pdb file.
FILE*   g_fRdf = NULL;                          // Radial distribution function file.
bool    g_bResult = true;                       // Whether to create the output file.
bool    g_bPdb = true;
bool    g_bRdf = true;

const char *g_szInpFile = TEXT("sugr_in.txt");       // Input file.

// Backup variables.
const char*  gszBckup[] = {TEXT("bckup0.sugr"),TEXT("bckup1.sugr")};// Note 2 backup files.
bool   gbBckup = false;                         // Whether to use backup.
bool   gbStartBckup = false;                    // Whether to start from backup file.
int    g_hstepBckup = 10000;                    // How often backup file is created.

// Interface variables.
int    giRegime = BULK;                         // Regime of simulation.

// Materials.
const char*  g_szMaterial[] = {TEXT("Copper (Cu)"), TEXT("Silver (Ag)"), TEXT("Gold (Au)"),
                               TEXT("Nickel (Ni)"), TEXT("Aluminium (Al)"), TEXT("Lead (Pb)")};      // materials
int    giMaterial = 3;         // Index of a material in the array of structures, Ni default.

// Many globals are taken from the input file, they provide communications
// between UI, structure g_hSimParams and device structure dparams in constant memory.
// Compute variables, "g_h" prefix - global on host.
real    g_hvelMag = 0.f;            // Magnitude of initial velocity, is defined in SetParams.
real    g_hrNebrShell = 0.4f;       // Distance to add to the cutoff for neighbors.
int     g_hiNebrMax = 2*BLOCK_SIZE; // Maximum number of neighbors.
int     g_hBlockSize = BLOCK_SIZE;  // Is used in shared memory arrays.
real    g_hTemperature = 298;       // Temperature, K (or heating temperature for CM).
int     g_hStepLimit = 50000;       // Duration of the simulation.
int     g_hrandSeedP;               // Seed for random numbers.
int     g_hstepAvg = 250;           // Step for averaging of measurements.
int     g_hstepPdb = 500;           // Step for printing of coordinates in pdb file.
int     g_hstepEquil = 50000;       // Equilibration period.
int     g_hstepCool = 5000;         // How long thermostat is applied after stepEquil.
real    g_hdeltaF = 0.0001;         // Increment of shear force to each atom in pN.
real    g_hdeltaT = 0.001;          // Time step.
int     g_hstepThermostat = 25;     // Step for applying Berendsen thermostat.
real    g_hgammaBerendsen = 0.005;  // Friction coefficient in Berendsen thermostat.
int     g_hcellShiftZ = 4;          // Number of cells under the graphene layer.
VecR    g_hregion;                  // Dimension of the simulation cell.

// Parameters, default values.
constexpr real    g_ha0 = 1.42;         // Interatomic distance in graphene, angstrom.
VecI    g_hinitUcell = {7, 7, 7};       // Number of unit cells of the surface.
real    g_hepsilon = 0.0087381;         // Lennard-Jones parameter, eV.
real    g_hsigma = 2.4945;              // Lennard-Jones parameter, angstrom.

// Deposition parameters.
real    g_hDeposEnergy = 0.03;      // Energy of deposited atoms eV.
int     g_hNmolDeposMe = 300;       // Number of deposited metallic atoms (used in SG).
int     g_hstepDeposit = 11000;     // How often to deposit atoms (every stepDeposit steps).
int     g_hnMolToDeposit = 40;      // Atoms that are simulteneously generated.

// rdf variables.
real    g_hrangeRdf = 4.f;          // Maximum distance between atoms for rdf.
int     g_hlimitRdf = 100;          // Number of measurements.
int     g_hsizeHistRdf = 200;       // Number of intervals in the histogram.
int     g_hstepRdf = 50;            // How often to make measurements of rdf.

// Diffusion variables.
int g_hnValDiffuse = 25000;
int g_hnBuffDiffuse = 10;
int g_hstepDiffuse = 100;
int g_hlimitDiffuseAv = 20;
int g_hcountDiffuseAv = 0;

// Contact mechanics.
uint3 g_unitCellMe;             // Number of unit cells in each direction for the metal slab.
real g_extrusion{ 0.0f };       // Substrate extrusion (same for all the directions), e.g. substrate width = metal_slab_width * (1 + extrusion)
real g_height_extrusion{ 0.f }; // Extrusion of the vertical region size as a fraction of the initial metal slab height.
real g_finalTemperature{ 298 }; // Final (cooling) temperature, K.
real g_maxNPHeightFraction{ 0.0f }; // Heating of NP happens until the NP's height is < originalNPHeight * (1. + g_maxNPHeightFraction).
int  g_coolingStepThermostat{ g_hstepThermostat }; // How often thermostat is applied during the cooling phase.

// Host variables.
float3  g_hvSum;                           // Total impulse.
SimParams g_hSimParams;                    // Is used for communication between UI and wrappers (and kernels).
float4  *g_hr = nullptr;                   // Host array for positions, note float4.
float4  *g_hspecForcesAndEnergy = nullptr; // Host, forces and potential energy of atoms of one type (e.g. metal) from another type (e.g. carbon).
float3  *g_hv = nullptr, *g_ha = nullptr;  // Host arrays for velocities and forces of molecules.

// Device variables.
float4  *g_dr = nullptr;                   // Device array for positions.

/////////////////////////
// Program's entry point.
/////////////////////////
int main(int argc, char* argv[])
{
    int deviceCount = 0;

    // Exit if there are no CUDA capable devices.
    gpuErrchk(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0){                  // Check the number of cuda devices.
        printf(TEXT("There are no CUDA capable devices!\n"));
        return 1;   // Exit the application.
    }

    // Initialize cuda.
    CudaInitW(argc, (char**)&g_szCmdLine);

    // Get cuda device properties.
    gpuErrchk(cudaGetDeviceProperties(&g_hDeviceProp, 0));

    std::cout << std::endl;
    std::cout << "GPU name =" << std::string(g_hDeviceProp.name) << std::endl;
    std::cout << "Compute capability = " << g_hDeviceProp.major << "." << g_hDeviceProp.minor << std::endl;
    std::cout << "Max threads per block = " << g_hDeviceProp.maxThreadsPerBlock << ", ";
    std::cout << "maxThreadsDim = " << g_hDeviceProp.maxThreadsDim[0] << " "
        << g_hDeviceProp.maxThreadsDim[1] << " " << g_hDeviceProp.maxThreadsDim[2] << ", ";
    std::cout << "maxGridSize = " << g_hDeviceProp.maxGridSize[0] << " "
        << g_hDeviceProp.maxGridSize[1] << " " << g_hDeviceProp.maxGridSize[2] << std::endl;
    std::cout << "Total global memory = " << g_hDeviceProp.totalGlobalMem <<
                 ", Total constant memroy = " << g_hDeviceProp.totalConstMem <<
                 ", Memory bus width = " << g_hDeviceProp.memoryBusWidth << std::endl;

    // Check compute capability, if less than 1.2 then exit.
    if ( (g_hDeviceProp.major < 1) ||
       ( (g_hDeviceProp.major == 1) && (g_hDeviceProp.minor < 2) ) )
    {
      printf(TEXT("Compute capability of your device is less than 1.2!"));
      return 1;                       // Exit the application.
    }

    // Read input parameters.
    if (ReadInputFile(g_szInpFile) == false) {
        printf(TEXT("Error with input file! Check it!\n"));
        return 1;
    }

    // Set host parameters and check system size.
    if (!SetParams()){
        printf(TEXT("The system is too large (grid = %i blocks)!\n"), g_hSimParams.gridSize);
        printf(TEXT("The application cannot be launched! Exit!\n"));
        return 1;   // Exit the application.
    }

    PrintParams();

    // Make preliminary work.
    if (!SetupJob()){
        printf(TEXT("Problems with SetupJob!\n"));
        return 1; // If error with initial coordinates then exit.
    }

    // Begin computations.
    printf(TEXT("\nPerforming computations. Wait...\n"));
    const char* errorString =
        DoComputationsW(g_hr, g_hv, g_ha, g_hspecForcesAndEnergy, &g_hSimParams, g_fResult, g_szPdb);

    if (errorString != 0) {
        printf(TEXT("Problems with DoComputationsW! We exit!\n"));
        return 1;
    }

    FreeArrays();       // Free memory and close files.

    printf(TEXT("Done!\n"));

    return 0;
}

/////////////////////////////////////
// Global functions implementations.
/////////////////////////////////////

void AllocArrays () // Allocate host memory.
{
    AllocMem(g_hr, g_hSimParams.nMol, float4);
    AllocMem(g_hv, g_hSimParams.nMol, float3);
    AllocMem(g_ha, g_hSimParams.nMol, float3);
    AllocMem(g_hspecForcesAndEnergy, g_hSimParams.nMol, float4);
}

void FreeArrays ()  // Free global memory on host and close files.
{
    if (g_hr)
        free(g_hr);
    if (g_hv)
        free(g_hv);
    if (g_ha)
        free(g_ha);

    if (g_hspecForcesAndEnergy)
        free(g_hspecForcesAndEnergy);

    if (g_fResult)
        fclose(g_fResult);
}

// Must be called before SetupJob.
int SetParams()
{
    // Define units of measurements.
    g_hSimParams.temperatureU = 298/0.409246;
    g_hSimParams.enU = 0.06275049;
    g_hSimParams.kB = 0.99999793;
    g_hSimParams.lengthU = g_ha0;
    g_hSimParams.forceU = 14.1239836;       // 1 nN dimensionless.
    g_hSimParams.massU = 12.0107;           // Carbon mass in amu.

    g_hSimParams.iRegime = giRegime;
    g_hSimParams.iMaterial = giMaterial;
    lstrcpy(g_hSimParams.szNameMe, Material[giMaterial].szName);

    // Check values of number of cells.
    if(g_hinitUcell.x < 2)
        g_hinitUcell.x = 2;
    if(g_hinitUcell.y < 2)
        g_hinitUcell.y = 2;

    EamInit();          // Initialize eam parameters.
    GrapheneInit();

    CalculateNumberOfAtoms(); // Must be called after EamInit() to have valid g_hSimParams.a

    CalculateRegionSize();

    // Save data in SimParams structure.
    VCopy(g_hSimParams.initUcell, g_hinitUcell);
    VCopy(g_hSimParams.unitCellMe, g_unitCellMe);
    VCopy(g_hSimParams.region, g_hregion);
    // This is inverse width of a cell.
    VDiv (g_hSimParams.invWidth, g_hSimParams.cells, g_hSimParams.region);
    g_hSimParams.cellShiftZ = g_hcellShiftZ;

    g_hSimParams.z_0 = 0.0f;

    // Save initial z coordinate of carbon atoms.
    if ((g_hSimParams.iRegime == SURFACE_GROWTH || g_hSimParams.iRegime == SHEAR) && g_hSimParams.nMolMe != 0)
    {
        g_hSimParams.z_0 =
            -0.5f * g_hSimParams.region.z + g_hSimParams.cellShiftZ / g_hSimParams.invWidth.z;
    }
    else if (g_hSimParams.iRegime == CONTACT_MECHANICS)
    {
        // Graphene initial coordinate is just g_hcellShiftZ * g_hSimParams.a above the bottom edge of the simulation box.
        g_hSimParams.z_0 = -0.5f * g_hSimParams.region.z + g_hSimParams.cellShiftZ * g_hSimParams.a;
    }

    // Dimensionless temperature.
    if(g_hTemperature <=0)
        g_hTemperature = 298;
    g_hSimParams.temperature = g_hTemperature / g_hSimParams.temperatureU;

    g_hSimParams.finalTemperature = g_finalTemperature / g_hSimParams.temperatureU;
    g_hSimParams.maxNPHeightFraction = g_maxNPHeightFraction;
    g_hSimParams.coolingStepThermostat = g_coolingStepThermostat;

    if(g_hstepAvg <=0)
        g_hstepAvg = 1;

    g_hSimParams.stepLimit = g_hStepLimit;
    g_hSimParams.stepAvg = g_hstepAvg;
    g_hSimParams.stepEquil = g_hstepEquil;
    g_hSimParams.stepPdb = g_hstepPdb;
    g_hSimParams.moreCycles = 1;
    g_hSimParams.deltaT = g_hdeltaT;
    g_hSimParams.rNebrShell = g_hrNebrShell;

    g_hSimParams.blockSize = g_hBlockSize;

    // This is the size of CELL array, it takes into account that there can be
    // blockSize atoms per cell.
    g_hSimParams.maxMol =
        g_hSimParams.cells.x * g_hSimParams.cells.y * g_hSimParams.cells.z *
        g_hSimParams.blockSize;

    // Number of blocks is equal to the number of cells.
    g_hSimParams.gridSize = g_hSimParams.cells.x * g_hSimParams.cells.y * g_hSimParams.cells.z;
    // If grid size is larger than device capabilities, then exit.
    if( g_hSimParams.gridSize > (uint)g_hDeviceProp.maxGridSize[0] )
        return 0;

    g_hSimParams.rrNebr = Sqr (g_hSimParams.rCutEam + g_hSimParams.rNebrShell);
    g_hSimParams.iNebrMax = g_hiNebrMax;    // Maximum number of neighbors per atom.

    g_hSimParams.bPdb = g_bPdb;
    g_hSimParams.bResult = g_bResult;

    // Thermostat.
    g_hSimParams.stepThermostat = g_hstepThermostat;
    g_hSimParams.gammaBerendsen = g_hgammaBerendsen;

    // For surface growth stepCool == stepLimit.
    if(g_hSimParams.iRegime == SURFACE_GROWTH)
        g_hSimParams.stepCool = g_hSimParams.stepLimit;
    else
        g_hSimParams.stepCool = g_hstepCool;

    // Increment of shear force
    // dimensionless = pN*unit[nN]/1000
    g_hSimParams.deltaF = g_hdeltaF*g_hSimParams.forceU*0.001;
    g_hSimParams.shear = 0.f;

    // Deposition.
    // Set deposition velocity.
    g_hSimParams.velMagDepos =
        sqrt(2*g_hDeposEnergy / (g_hSimParams.enU * g_hSimParams.massMe));
    g_hSimParams.stepDeposit = g_hstepDeposit;
    g_hSimParams.nMolToDeposit = g_hnMolToDeposit;
    // If not SG then nMolDeposited == nMolMe.
    if(g_hSimParams.iRegime == SURFACE_GROWTH)
        g_hSimParams.nMolDeposited = 0;
    else
        g_hSimParams.nMolDeposited = g_hSimParams.nMolMe;

    // Magnitude of initial velocities.
    if (g_hSimParams.iRegime == BULK)                      // For metal use mass.
    {
        g_hvelMag = sqrt(NDIM * (1. - 1. / g_hSimParams.nMol) *
            g_hSimParams.kB * g_hSimParams.temperature / g_hSimParams.massMe);
    }
    else if (g_hSimParams.iRegime == SURFACE_GROWTH || g_hSimParams.iRegime == SHEAR)
    {
        g_hvelMag = sqrt(NDIM * (1. - 1. / (g_hSimParams.nMol))
            * g_hSimParams.kB * g_hSimParams.temperature *
            g_hSimParams.nMol / (g_hSimParams.nMol - g_hSimParams.nMolMe)); // Carbon mass == 1.
    }
    else if (g_hSimParams.iRegime == CONTACT_MECHANICS)
    {
        // Take into account both types of atoms.
        g_hvelMag = sqrt(NDIM * (1. - 1. / g_hSimParams.nMol) *
            g_hSimParams.kB * g_hSimParams.temperature / (g_hSimParams.massMe + 1.)); // Carbon mass == 1.
    }

    g_hSimParams.velMag = g_hvelMag;

    // rdf variables.
    g_hSimParams.rangeRdf = 1.5*g_hSimParams.rCutEam; //g_hrangeRdf;
    g_hSimParams.limitRdf = g_hlimitRdf;
    g_hSimParams.sizeHistRdf = g_hsizeHistRdf;
    g_hSimParams.stepRdf = g_hstepRdf;
    g_hSimParams.countRdf = 0;
    g_hSimParams.intervalRdf = g_hSimParams.sizeHistRdf / g_hSimParams.rangeRdf;
    g_hSimParams.bRdf = g_bRdf;
    lstrcpy(g_hSimParams.szRdfPath, g_szRdf);

    // Backup.
    lstrcpy(g_hSimParams.szBckup0, gszBckup[0]);
    lstrcpy(g_hSimParams.szBckup1, gszBckup[1]);
    g_hSimParams.bBckup = gbBckup;
    g_hSimParams.bStartBckup = gbStartBckup;
    g_hSimParams.stepBckup = g_hstepBckup;
    // Zero duration of the simulation.
    g_hSimParams.totalTime= 0.f;

    // Diffusion variables.
    g_hSimParams.nValDiffuse = g_hnValDiffuse;
    g_hSimParams.nBuffDiffuse = g_hnBuffDiffuse;
    g_hSimParams.stepDiffuse = g_hstepDiffuse;
    g_hSimParams.limitDiffuseAv = g_hlimitDiffuseAv;
    g_hSimParams.countDiffuseAv = g_hcountDiffuseAv;

    // Set seed for random numbers, it is in params structure.
    InitRand(0, &g_hSimParams);
    // Copy parameters to the device, global variable dparams in const mem is initialized.
    SetParametersW(&g_hSimParams);          // Wrapper from .cu file.

    return 1;       // All right.
}

void PrintParams()
{
    std::cout << std::endl;
    std::cout << "Input parameters:" << std::endl;
    std::cout << "Regime = " << g_hSimParams.iRegime << std::endl;
    std::cout << "Material = " << g_hSimParams.iMaterial << std::endl;
    std::cout << "Total atoms = " << g_hSimParams.nMol << std::endl;
    std::cout << "Metal atoms = " << g_hSimParams.nMolMe << std::endl;

    std::cout << "Carb cells = " << g_hSimParams.initUcell.x << " " << g_hSimParams.initUcell.y
              << " " << g_hSimParams.initUcell.z << std::endl;

    std::cout << "lattice a = " << g_hSimParams.a * g_hSimParams.lengthU << " Angstrom" << std::endl;
    std::cout << "sigmaLJ = " << g_hSimParams.sigmaLJ * g_hSimParams.lengthU << " Angstrom" << std::endl;
    std::cout << "epsLJ = " << g_hSimParams.epsLJ * g_hSimParams.enU << " eV" << std::endl;
    std::cout << "rCutLJ = " << g_hSimParams.rCutLJ * g_hSimParams.lengthU << " Angstrom" << std::endl;

    std::cout << "T = " << g_hSimParams.temperature * g_hSimParams.temperatureU << " K" << std::endl;

    std::cout << "stepLimit = " << g_hSimParams.stepLimit << std::endl;
    std::cout << "stepEquil = " << g_hSimParams.stepEquil << std::endl;
    std::cout << "stepCool = " << g_hSimParams.stepCool << std::endl;
    std::cout << "stepThermostat = " << g_hSimParams.stepThermostat << std::endl;
    std::cout << "stepRdf = " << g_hSimParams.stepRdf << std::endl;
    std::cout << "stepPdb = " << g_hSimParams.stepPdb << std::endl;

    std::cout << "randSeed = " << g_hSimParams.randSeedP << std::endl;
}

// Make preliminary work.
int SetupJob()
{
    AllocArrays();  // Allocate host globals.
    g_hSimParams.stepCount = 0;

    // Compute initial coordinates using kernel.
    const char* errorString = InitCoordsW(g_dr, g_hr, &g_hSimParams);   // Wrapper from .cu file.
    // Check errors.
    if(errorString != 0) {
        char szBuf[MAX_PATH] = TEXT("Exception: ");
        lstrcat(szBuf, errorString);
        printf(TEXT("%s\n"), szBuf);
        printf(TEXT("Problems with initial coordinates! We exit!\n"));
        return 0;
    }

    InitVels();                         // Initialize velocities on host.
    InitAccels();                       // Initialize accelerations on host.
    InitSpecForcesAndEnergy();
    AccumProps(0, &g_hSimParams);       // Zero properties.
    g_hSimParams.nebrNow = 1;           // Neighbor list should be built.

    // Open file.
    if(g_bResult) {
        char szBuf[MAX_PATH];
        // Define file name depending on the regime.
        if(g_hSimParams.iRegime == BULK)
        sprintf(szBuf, TEXT("blk_%s_x%i_y%i_z%i_Me%i_Av%i_Pdb%i_T%3.0f"),
            g_hSimParams.szNameMe, g_hSimParams.initUcell.x, g_hSimParams.initUcell.y,
            g_hSimParams.initUcell.z,
            g_hSimParams.nMolMe,
            g_hSimParams.stepAvg, g_hSimParams.stepPdb, g_hSimParams.temperature
            *g_hSimParams.temperatureU);

        if(g_hSimParams.iRegime == SURFACE_GROWTH)   // If surface growth.
            sprintf(szBuf, TEXT("sg_%s_x%i_y%i_Me%i_Eq%i_Dep%i_TD%i_Pdb%i_T%3.0f"),
            g_hSimParams.szNameMe, g_hSimParams.initUcell.x, g_hSimParams.initUcell.y,
            g_hSimParams.nMolMe, g_hSimParams.stepEquil, g_hSimParams.stepDeposit,
            g_hSimParams.nMolToDeposit, g_hSimParams.stepPdb, g_hSimParams.temperature
            *g_hSimParams.temperatureU);

        if(g_hSimParams.iRegime == SHEAR)   // If shear.
            sprintf(szBuf, TEXT("sh_%s_x%i_y%i_Me%i_Eq%i_C%i_Av%i_Pdb%i_T%3.0f"),
            g_hSimParams.szNameMe, g_hSimParams.initUcell.x, g_hSimParams.initUcell.y,
            g_hSimParams.nMolMe, g_hSimParams.stepEquil, g_hSimParams.stepCool,
            g_hSimParams.stepAvg, g_hSimParams.stepPdb,
            g_hSimParams.temperature*g_hSimParams.temperatureU);

        if (g_hSimParams.iRegime == CONTACT_MECHANICS)
            sprintf(szBuf, TEXT("cm_%s_x%i_y%i_z%i_Me%i_cx%i_cy%i_Av%i_Pdb%i_T%3.0f_lim%i"),
            g_hSimParams.szNameMe, g_unitCellMe.x, g_unitCellMe.y, g_unitCellMe.z,
            g_hSimParams.nMolMe, g_hSimParams.initUcell.x, g_hSimParams.initUcell.y,
            g_hSimParams.stepAvg, g_hSimParams.stepPdb,
            g_hSimParams.temperature * g_hSimParams.temperatureU, g_hSimParams.stepLimit);

        lstrcat(szBuf, TEXT(".txt"));
        lstrcat(g_szResult, szBuf);

        g_fResult = fopen(g_szResult, TEXT("w"));
        fprintf(g_fResult,
TEXT("stepCnt impulse totEn(eV) totEn.rms(eV) potEn(eV) potEn.rms(eV) Tempr(K) T.rms(K) oneStep(ms) Veloc_CM CM(angstr) friction(nN) sizex(angstr) sizey(angstr) sizez(angstr) shearForce(nN) \n"));

        // Print additional values to cmd.
        if( g_hSimParams.bResult != 0 )
            printf ("time step = %f", g_hSimParams.deltaT);
        if( (g_hSimParams.bResult != 0) && (g_hSimParams.iRegime == SURFACE_GROWTH) )
            printf ("deposit energy = %f eV ", g_hDeposEnergy);
        // Print increment of shear force.
        if( (g_hSimParams.bResult != 0) && (g_hSimParams.iRegime == SHEAR) )
            printf ("increment of shear = %f pN ", g_hSimParams.deltaF*1000/g_hSimParams.forceU);

        if( (g_hSimParams.bResult != 0) && (g_hSimParams.iRegime == CONTACT_MECHANICS) )
        {
            std::cout << "\nmetal cells = " << g_hSimParams.unitCellMe.x << " " << g_hSimParams.unitCellMe.y
                      << " " << g_hSimParams.unitCellMe.z << std::endl;
            std::cout << "height_extrusion = " << g_height_extrusion << std::endl;
            std::cout << "substrate_extrusion = " << g_extrusion << std::endl;
            std::cout << "final T = " << g_hSimParams.finalTemperature * g_hSimParams.temperatureU << std::endl;
            std::cout << "Height fraction = " << g_hSimParams.maxNPHeightFraction << std::endl;
            std::cout << "Thermostat cooling step = " << g_hSimParams.coolingStepThermostat << std::endl;
        }

        if( g_hSimParams.bResult != 0 )
            printf ("\n");
    }       // End if(g_bResult).

    if(g_hSimParams.iRegime == SHEAR) { // If shear specify diffuse path.
        lstrcpy(g_hSimParams.szDiffusePath, TEXT("Diffuse.txt"));
    }

    return 1;
}

// Initialize parameters for EAM potential.
void EamInit()
{
    // Define dimensionless parameters for EAM potential from Zhou et al.
    // double eV = 1.60219 / pow(10.0, 19.0);       // 1 electron - volt
    // unit of length = 1.42 angstrom, so dimensionless length = length (angsrom) / 1.42
    // energy unit 6.275049 * 10^(-2)eV

    // Parameters for metal.
    g_hSimParams.re = Material[g_hSimParams.iMaterial].re / g_ha0;// Nearest neighbors.

    g_hSimParams.a = g_hSimParams.re*sqrt(2.);      // elementary translation / g_ha0;

    g_hSimParams.fe = Material[g_hSimParams.iMaterial].fe;
    g_hSimParams.rhoe = Material[g_hSimParams.iMaterial].rhoe;
    g_hSimParams.alpha = Material[g_hSimParams.iMaterial].alpha;
    g_hSimParams.beta = Material[g_hSimParams.iMaterial].beta;

    g_hSimParams.A = Material[g_hSimParams.iMaterial].A / g_hSimParams.enU;
    g_hSimParams.B = Material[g_hSimParams.iMaterial].B / g_hSimParams.enU;

    g_hSimParams.kappa = Material[g_hSimParams.iMaterial].kappa;
    g_hSimParams.lambda = Material[g_hSimParams.iMaterial].lambda;

    g_hSimParams.Fn[0] = Material[g_hSimParams.iMaterial].Fn[0] /g_hSimParams.enU;
    g_hSimParams.Fn[1] = Material[g_hSimParams.iMaterial].Fn[1] /g_hSimParams.enU;
    g_hSimParams.Fn[2] = Material[g_hSimParams.iMaterial].Fn[2] /g_hSimParams.enU;
    g_hSimParams.Fn[3] = Material[g_hSimParams.iMaterial].Fn[3] /g_hSimParams.enU;

    g_hSimParams.F[0] = Material[g_hSimParams.iMaterial].F[0] /g_hSimParams.enU;
    g_hSimParams.F[1] = Material[g_hSimParams.iMaterial].F[1] /g_hSimParams.enU;
    g_hSimParams.F[2] = Material[g_hSimParams.iMaterial].F[2] /g_hSimParams.enU;
    g_hSimParams.F[3] = Material[g_hSimParams.iMaterial].F[3] /g_hSimParams.enU;

    g_hSimParams.eta = Material[g_hSimParams.iMaterial].eta;

    g_hSimParams.Fe = Material[g_hSimParams.iMaterial].Fe /g_hSimParams.enU;

    g_hSimParams.massMe = Material[g_hSimParams.iMaterial].massMe / g_hSimParams.massU; // Dimensionless nickel mass
    // Dimensionless density == density * lengthU^3 [m^3] / massU[kg]
    g_hSimParams.density = Material[g_hSimParams.iMaterial].density * 0.1435655 * 0.001; //1.27802;

    // Set cutoff distance, it defines number of BLOCK_SIZE,
    // e.g. for 1.5 BLOCK_SIZE should be 96 for lead, but for Ni it can be 48.
    g_hSimParams.rCutEam = 1.45 * g_hSimParams.a;
    g_hSimParams.rrCutEam = g_hSimParams.rCutEam * g_hSimParams.rCutEam;

    // Inverse values.
    g_hSimParams.rei = 1 / g_hSimParams.re;
    g_hSimParams.rhoei = 1 / g_hSimParams.rhoe;
}

void GrapheneInit()
{
    // Define dimensionless parameters for graphite potential from Sasaki et al.
    g_hSimParams.mu_r = 41.881*g_ha0*g_ha0 / g_hSimParams.enU;      // 41.881 eV / angsrom^2
    g_hSimParams.theta_0 = 2.*M_PI/3.;                              // 2*M_PI/3, dimensionless
    g_hSimParams.mu_theta = 2.9959*g_ha0*g_ha0 / g_hSimParams.enU;  // 2.9959 eV / angstrom^2
    g_hSimParams.mu_p = 18.225*g_ha0*g_ha0 / g_hSimParams.enU;      // 18.225 eV / angstrom^2
    g_hSimParams.rCutC = 1.2;                                   // cutoff = 1.42*1.2 angstrom
    g_hSimParams.rrCutC = g_hSimParams.rCutC * g_hSimParams.rCutC;
    // LJ parameters.
    g_hSimParams.epsLJ = g_hepsilon / g_hSimParams.enU;             // 0.87381*10^(-2) eV
    g_hSimParams.sigmaLJ = g_hsigma / g_ha0;                        // 2.4945 angstrom
    g_hSimParams.rCutLJ = 2.5*g_hSimParams.sigmaLJ;
    g_hSimParams.rrCutLJ = g_hSimParams.rCutLJ*g_hSimParams.rCutLJ;
    g_hSimParams.initSlabHeight = 0.5 * g_hSimParams.sigmaLJ;      // Initial height related to the LJ energy minimum distance
}

void CalculateNumberOfAtoms()
{
    if (g_hSimParams.iRegime == BULK)      // If bulk then all atoms are metal.
    {
        g_hSimParams.nMolMe = 4 * g_hinitUcell.x * g_hinitUcell.y * g_hinitUcell.z;
        g_hSimParams.nMol = g_hSimParams.nMolMe;
    }
    // If surface growth or shear, first nMolMe atoms are Me, from nMolMe to nMol are carbon atoms.
    else if (g_hSimParams.iRegime == SURFACE_GROWTH || g_hSimParams.iRegime == SHEAR)
    {
        g_hSimParams.nMolMe = g_hNmolDeposMe;
        g_hSimParams.nMol = g_hSimParams.nMolMe + 32 * g_hinitUcell.x * g_hinitUcell.y;
        g_hinitUcell.z = 0;
    }
    else if (g_hSimParams.iRegime == CONTACT_MECHANICS)
    {
        // Metal atoms are arranged in a FCC prysm (4 atoms per unit cell).
        g_hSimParams.nMolMe = 4 * g_unitCellMe.x * g_unitCellMe.y * g_unitCellMe.z;

        const auto dx_me = g_unitCellMe.x * g_hSimParams.a;
        const auto dy_me = g_unitCellMe.y * g_hSimParams.a;
        const auto dz_me = g_unitCellMe.z * g_hSimParams.a;

        constexpr real cos30 = 0.8660254038f;
        g_hinitUcell.x = ceil(dx_me * (1. + g_extrusion) / (8. * cos30)); // Carbon covalent bond length is 1.0 dimensionless.
        g_hinitUcell.y = ceil(dy_me * (1. + g_extrusion) / 6.); // Carbon covalent bond length is 1.0 dimensionless.
        g_hinitUcell.z = 0;

        g_hSimParams.nMol = g_hSimParams.nMolMe + 32 * g_hinitUcell.x * g_hinitUcell.y;
    }
}

void CalculateRegionSize()
{
    if (g_hSimParams.iRegime == BULK)
    {
        VSCopy(g_hregion, 1. / pow((g_hSimParams.density * 0.25 / g_hSimParams.massMe), 1. / 3.),
            g_hinitUcell);
        VSCopy(g_hSimParams.cells, 1.0 / (g_hSimParams.rCutEam + g_hrNebrShell), g_hregion);
    }
    else if (g_hSimParams.iRegime != BULK) // If surface growth/shear/contact mechanics.
    {
        g_hregion.x = g_hinitUcell.x * 8 * cos(M_PI / 6.);//*a == 1
        g_hregion.y = g_hinitUcell.y * 6;                 //*a == 1

        // If surface growth region.z is larger than for shear.
        if (g_hSimParams.iRegime == SURFACE_GROWTH)
        {
            // Define number of cells under the graphene layer.
            if (g_hSimParams.nMolMe == 0) g_hcellShiftZ = 0;    // Num of cells under graphene.
            else if ((g_hSimParams.nMolMe > 0) && (g_hSimParams.nMolMe < 100)) g_hcellShiftZ = 1;
            else if ((g_hSimParams.nMolMe >= 100) && (g_hSimParams.nMolMe < 5000)) g_hcellShiftZ = 2;
            else if ((g_hSimParams.nMolMe >= 5000) && (g_hSimParams.nMolMe < 10000)) g_hcellShiftZ = 3;
            else if ((g_hSimParams.nMolMe >= 10000) && (g_hSimParams.nMolMe < 20000)) g_hcellShiftZ = 4;
            else if (g_hSimParams.nMolMe >= 20000) g_hcellShiftZ = 5;

            // region.z = 1.8*height of the cube + a + number of cells under graphene.
            if (g_hSimParams.nMolMe != 0)
                g_hregion.z = 1.8 * g_hSimParams.a * pow((0.25 * g_hSimParams.nMolMe), 0.33333333333) +
                g_hSimParams.a + g_hcellShiftZ * (g_hSimParams.rCutEam + g_hrNebrShell);
            else    // If no metal atoms, then simply several cells.
                g_hregion.z = 3 * (g_hSimParams.rCutEam + g_hrNebrShell);
        }
        // If shear.
        else if (g_hSimParams.iRegime == SHEAR)
        {
            // Define number of cells under the graphene layer.
            if (g_hSimParams.nMolMe == 0) g_hcellShiftZ = 0;    // Num of cells under graphene.
            else if ((g_hSimParams.nMolMe > 0) && (g_hSimParams.nMolMe < 100)) g_hcellShiftZ = 1;
            else if ((g_hSimParams.nMolMe >= 100) && (g_hSimParams.nMolMe < 1000)) g_hcellShiftZ = 2;
            else if ((g_hSimParams.nMolMe >= 1000) && (g_hSimParams.nMolMe < 5000)) g_hcellShiftZ = 3;
            else if ((g_hSimParams.nMolMe >= 5000) && (g_hSimParams.nMolMe < 10000)) g_hcellShiftZ = 5;
            else if ((g_hSimParams.nMolMe >= 10000) && (g_hSimParams.nMolMe < 20000)) g_hcellShiftZ = 7;
            else if (g_hSimParams.nMolMe >= 20000) g_hcellShiftZ = 9;

            // region.z = height of the cube + 3.6*a + number of cells under graphene.
            if (g_hSimParams.nMolMe != 0)
                g_hregion.z = g_hSimParams.a * pow((0.25 * g_hSimParams.nMolMe), 0.33333333333) +
                3.6 * g_hSimParams.a + g_hcellShiftZ * (g_hSimParams.rCutEam + g_hrNebrShell);
            else    // If no metal atoms, then simply several cells.
                g_hregion.z = 3 * (g_hSimParams.rCutEam + g_hrNebrShell);
        }
        else if (g_hSimParams.iRegime == CONTACT_MECHANICS)
        {
            // Define number of cells under the graphene layer.
            if ((g_hSimParams.nMolMe > 0) && (g_hSimParams.nMolMe < 100)) g_hcellShiftZ = 1;
            else if ((g_hSimParams.nMolMe >= 100) && (g_hSimParams.nMolMe < 1000)) g_hcellShiftZ = 2;
            else if ((g_hSimParams.nMolMe >= 1000) && (g_hSimParams.nMolMe < 5000)) g_hcellShiftZ = 3;
            else if ((g_hSimParams.nMolMe >= 5000) && (g_hSimParams.nMolMe < 10000)) g_hcellShiftZ = 5;
            else if ((g_hSimParams.nMolMe >= 10000) && (g_hSimParams.nMolMe < 20000)) g_hcellShiftZ = 7;
            else if (g_hSimParams.nMolMe >= 20000) g_hcellShiftZ = 9;

            // region.z = height of the slab * (1 + P) + initialHeight + number of cells under graphene.
            g_hregion.z = g_unitCellMe.z * g_hSimParams.a * (1. + g_height_extrusion) + g_hSimParams.initSlabHeight
                + g_hcellShiftZ * g_hSimParams.a;
        }

        VSCopy(g_hSimParams.cells, 1.0 / (g_hSimParams.rCutEam + g_hrNebrShell), g_hregion);
    }
}

// Code for generation of random numbers (from Rapaport).
void InitRand (int randSeedI, SimParams *hparams)
{
    struct tm tv {};

    if (randSeedI != 0)
    {
        std::cout << "Non-random seed " << randSeedI << " is used." << std::endl;
        hparams->randSeedP = randSeedI;
    }
    else {
        std::mt19937 engine(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        constexpr auto minVal = 1;
        constexpr auto maxVal = 1000;
        std::uniform_int_distribution<int> dist(minVal, maxVal);

        hparams->randSeedP = dist(engine);
    }
}

real RandR (SimParams *hparams)
{
    hparams->randSeedP = (hparams->randSeedP * IMUL + IADD) & MASK_RAND;
    return (hparams->randSeedP * SCALE);
}

void VRand (float3 *p, SimParams *hparams)
{
    real s = 0.;
    real x = 0.;
    real y = 0.;

    s = 2.;
    while (s > 1.) {
        x = 2. * RandR (hparams) - 1.;
        y = 2. * RandR (hparams) - 1.;
        s = Sqr (x) + Sqr (y);
    }
    p->z = 1. - 2. * s;
    s = 2. * sqrt (1. - s);
    p->x = s * x;
    p->y = s * y;
}

// Initialize velocities.
void InitVels()
{
    int n = 0;

    VZero (g_hvSum);
    for(n = 0; n < g_hSimParams.nMol; n++)
    {
        // For surface growth and shear, metal atoms have zero initial velocity.
        if( ((g_hSimParams.iRegime == SURFACE_GROWTH)
            || (g_hSimParams.iRegime == SHEAR))
            && (n < g_hSimParams.nMolMe) ) {// For metal.
            g_hv[n].x = 0.;
            g_hv[n].y = 0.;
            g_hv[n].z = 0.;
        }
        else // Bulk, contact mechanics or carbon atoms in sg/shear.
        {
            VRand (&g_hv[n], &g_hSimParams);
            VScale (g_hv[n], g_hSimParams.velMag);
            VVAdd (g_hvSum, g_hv[n]);
        }
    }

    // Shift velocities to provide zero total impulse.
    for (n = 0; n < g_hSimParams.nMol; n++)
    {
        if (g_hSimParams.iRegime == BULK)
        {
            VVSAdd(g_hv[n], -1. / (g_hSimParams.nMol), g_hvSum);
        }
        else if (((g_hSimParams.iRegime == SURFACE_GROWTH)
            || (g_hSimParams.iRegime == SHEAR)) && (n >= g_hSimParams.nMolMe))
        {
            VVSAdd(g_hv[n], -1. / (g_hSimParams.nMol /*- g_hSimParams.nMolMe*/), g_hvSum);
        }
        else if (g_hSimParams.iRegime == CONTACT_MECHANICS)
        {
            // Sum of g_hv[i] is originally g_hvSum. We add -g_hvSum/n to each vel.
            // Then the sum is g_hvSum - n * g_hvSum/n = 0.
            VVSAdd(g_hv[n], -1. / (g_hSimParams.nMol), g_hvSum);
        }
    }
}

// Initialize accelerations.
void InitAccels ()
{
    for(int n = 0; n < g_hSimParams.nMol; n++)
        VZero (g_ha[n]);
}

void InitSpecForcesAndEnergy()
{
    for (int n = 0; n < g_hSimParams.nMol; n++)
    {
        g_hspecForcesAndEnergy[n].x = 0.0f;
        g_hspecForcesAndEnergy[n].y = 0.0f;
        g_hspecForcesAndEnergy[n].z = 0.0f;
        g_hspecForcesAndEnergy[n].w = 0.0f;
    }
}

// Read input file, returns 0 if there is some error.
bool ReadInputFile(const char *szInpFile)
{
    char c = 0, szBuf[MAX_PATH], szTmp[MAX_PATH];
    int iCount = 0;
    int iLocalCnt = 0;
    int iRowCount = 0;

    // Open files.
    FILE *file = fopen(szInpFile, TEXT("r"));
    if (file == NULL)
    {
        std::cout << "Failed to open file " << std::string(szInpFile) << std::endl;
        return false;
    }

    // Parse lines.
    iCount = 0;
    iRowCount = 0;
    while( !feof(file) )
    {
        // Read lines.
        c = (char)fgetc(file);
        if( c != TEXT('\n') ) {
            szBuf[iCount] = c;
            ++iCount;
            continue;
        }
        else
        {
            // Append indication of a string array.
            szBuf[iCount] = TEXT('\0');
            iCount = 0;

            // Skip blanks.
            iLocalCnt = 0;
            while(szBuf[iLocalCnt] == TEXT(' '))
                ++iLocalCnt;

            // Skip empty lines.
            if( szBuf[iLocalCnt] == TEXT('\0') )
                continue;

            // If the first character is # then skip the line.
            if( szBuf[iLocalCnt] == TEXT('#') )
                continue;

            // Enlarge counter of rows here - it corresponds to the number in the input file.
            ++iRowCount;

            // Parse the row. Space is a delimeter.
            iLocalCnt = 0;
            while( 1 ) {
                if( szBuf[iLocalCnt] == TEXT(' ') ) break;
                szTmp[iLocalCnt] = szBuf[iLocalCnt];
                ++iLocalCnt;
            }
            szTmp[iLocalCnt] = TEXT('\0');
            ++iLocalCnt;

            // Analyze what we have read.
            switch( iRowCount ) {
                case 1:
                    giRegime = atoi(szTmp); // Regime.
                    break;

                case 2:
                    giMaterial = atoi(szTmp);   // Metal.
                    break;

                case 3:
                    g_hinitUcell.x = atoi(szTmp); // x cells.
                    break;

                case 4:
                    g_hinitUcell.y = atoi(szTmp); // y cells.
                    break;

                case 5:
                    g_hinitUcell.z = atoi(szTmp); // z cells.
                    break;

                case 6:
                    g_hTemperature = atof(szTmp); // Temperature.
                    break;

                case 7:
                    g_hNmolDeposMe = atoi(szTmp); // Number of metal atoms.
                    break;

                case 8:
                    g_hepsilon = atof(szTmp); // Epsilon for LJ.
                    break;

                case 9:
                    g_hsigma = atof(szTmp); // Sigma for LJ.
                    break;

                case 10:
                    g_hStepLimit = atoi(szTmp); // Duration of the simulation.
                    break;

                case 11:
                    g_hdeltaT = atof(szTmp); // Time step.
                    break;

                case 12:
                    g_hstepAvg = atoi(szTmp); // Averaging interval.
                    break;

                case 13:
                    g_hstepEquil = atoi(szTmp); // Equilibration interval.
                    break;

                case 14:
                    g_hdeltaF = atof(szTmp); // Increment of shear force.
                    break;

                case 15:
                    g_hstepCool = atoi(szTmp); // Cooling interval (for shear).
                    break;

                case 16:
                    g_hDeposEnergy = atof(szTmp); // Energy of deposited atoms eV.
                    break;

                case 17:
                    g_hnMolToDeposit = atoi(szTmp); // Atoms that are simulteneously generated.
                    break;

                case 18:
                    g_hstepDeposit = atoi(szTmp); // How often to deposit atoms (every stepDeposit steps)
                    break;

                case 19:
                    g_hstepPdb = atoi(szTmp); // Create pdb file each such interval.
                    break;

                case 20:
                    g_hstepRdf = atoi(szTmp); // Compute rdf each such interval.
                    break;

                case 21:
                    gbBckup = atoi(szTmp); // Whether to create backup.
                    break;

                case 22:
                    gbStartBckup = atoi(szTmp); // Whether to start from backup file.
                    break;

                case 23:
                    g_unitCellMe.x = atoi(szTmp); // Metal cells count in x direction.
                    break;

                case 24:
                    g_unitCellMe.y = atoi(szTmp); // Metal cells count in y direction.
                    break;

                case 25:
                    g_unitCellMe.z = atoi(szTmp); // Metal cells count in z direction.
                    break;

                case 26:
                    g_extrusion = atof(szTmp); // Substrate extrusion.
                    break;

                case 27:
                    g_height_extrusion = atof(szTmp); // Vertical region extrusion as fraction of the NP initial height.
                    break;

                case 28:
                    g_hstepThermostat = atoi(szTmp); // Apply the thermostat every g_hstepThermostat steps (in CM only heating).
                    break;

                case 29:
                    g_finalTemperature = atof(szTmp); // Final temperature (typically cooling T after the NP has melted).
                    break;

                case 30:
                    g_maxNPHeightFraction = atof(szTmp); // Heating of NP happens until the NP's height is < originalNPHeight * (1. + g_maxNPHeightFraction).
                    break;

                case 31:
                    g_coolingStepThermostat = atoi(szTmp); // Frequency of thermostat during the cooling phase.
                    break;

            } // End switch row count.
        }// End else c == \n.
    }

    fclose(file);

    return true;
}