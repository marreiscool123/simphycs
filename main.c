/*
    ============================================================
                         SimPhycs Alpha
    ============================================================

    Physics / 3D sandbox editor

    Build:
        gcc main.c -o simphycs.exe -lraylib -lopengl32 -lgdi32 -lwinmm -lm

    CAMERA
        RMB                 Orbit
        MMB                 Pan
        Wheel               Zoom
        WASD                Camera movement
        Shift + WASD        Fast camera movement
        Space               Run / Stop

    TOOLS
        Q                   Select
        W                   Move
        E                   Rotate
        R                   Scale
        F                   Force

        X                   X axis
        Y                   Y axis
        Z                   Z axis
        C                   Duplicate
        Delete              Delete
        Escape              Deselect

    CREATE
        1                   Block
        2                   Ball
        3                   Cylinder
        4                   Base

    MOTORS
        M                   Spin
        P                   Push
        B                   Conveyor

    MOVE
        Drag                Free X/Z
        Y                   Y axis
        X                   X axis
        Z                   Z axis
        Mouse wheel         Vertical movement while dragging

    ROTATE
        Drag X              Rotate Y
        Drag Y              Rotate X
        X/Y/Z               Axis constraint

    SCALE
        Drag X              X scale
        Drag Y              Z scale
        Y                   Y scale
        Ctrl                Uniform
        Shift               Y scale

    FORCE
        Drag                Physical impulse
*/

#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#define MAX_PARTS 256
#define MAX_SPRINGS 128

#define WINDOW_WIDTH 1440
#define WINDOW_HEIGHT 900

#define PHYSICS_HZ 240.0f
#define PHYSICS_DT (1.0f / PHYSICS_HZ)

#define MAX_SUBSTEPS_PER_FRAME 12
#define SOLVER_ITERATIONS 10

#define GRAVITY -24.0f
#define MAX_VELOCITY 120.0f
#define MAX_ANGULAR_VELOCITY 35.0f

#define PI_F 3.14159265358979323846f
#define RAD2DEG_F 57.29577951308232f

/* ------------------------------------------------------------ */
/* Helpers                                                       */
/* ------------------------------------------------------------ */

static Vector2 V2(float x, float y)
{
    Vector2 v = { x, y };
    return v;
}

static Vector3 V3(float x, float y, float z)
{
    Vector3 v = { x, y, z };
    return v;
}

static float ClampF(float v, float min, float max)
{
    if (v < min) return min;
    if (v > max) return max;
    return v;
}

static float AbsF(float v)
{
    return fabsf(v);
}

static float Max3(float a, float b, float c)
{
    float result = a;

    if (b > result) result = b;
    if (c > result) result = c;

    return result;
}

static Vector3 SafeNormalize(Vector3 v)
{
    float length = Vector3Length(v);

    if (length < 0.000001f)
        return V3(0, 0, 0);

    return Vector3Scale(v, 1.0f / length);
}

static Vector3 AbsVector3(Vector3 v)
{
    return V3(
        fabsf(v.x),
        fabsf(v.y),
        fabsf(v.z)
    );
}

static float MaxComponent(Vector3 v)
{
    return Max3(
        fabsf(v.x),
        fabsf(v.y),
        fabsf(v.z)
    );
}

/* ------------------------------------------------------------ */
/* Types                                                         */
/* ------------------------------------------------------------ */

typedef enum
{
    PART_BLOCK,
    PART_BALL,
    PART_CYLINDER,
    PART_BASE
} PartType;

typedef enum
{
    TOOL_SELECT,
    TOOL_MOVE,
    TOOL_ROTATE,
    TOOL_SCALE,
    TOOL_FORCE
} ToolType;

typedef enum
{
    AXIS_FREE,
    AXIS_X,
    AXIS_Y,
    AXIS_Z
} AxisMode;

typedef enum
{
    MOTOR_NONE,
    MOTOR_SPIN,
    MOTOR_PUSH,
    MOTOR_CONVEYOR
} MotorType;

typedef struct
{
    bool used;

    char name[32];

    PartType type;

    Vector3 position;
    Vector3 rotation;
    Vector3 size;

    Vector3 velocity;
    Vector3 angularVelocity;

    float mass;
    float friction;
    float bounce;

    bool anchored;
    bool frozen;
    bool collisions;

    bool sleeping;
    float sleepTimer;

    MotorType motor;
    float motorPower;

    bool hasSpring;
    int springTarget;
    float springStrength;
    float springLength;

    Color color;

    Model model;
} Part;

typedef struct
{
    bool used;

    int a;
    int b;

    float restLength;
    float stiffness;
    float damping;
} Spring;

typedef struct
{
    bool valid;

    Part parts[MAX_PARTS];
    Spring springs[MAX_SPRINGS];

    int selectedPart;
    int objectCounter;
} SceneSnapshot;

typedef struct
{
    Vector3 center;

    Vector3 half;

    Vector3 axis[3];
} OBB;

typedef struct
{
    Camera3D camera;

    Vector3 target;

    float yaw;
    float pitch;
    float distance;

    bool orbit;
    bool pan;
} EditorCamera;

/* ------------------------------------------------------------ */
/* Globals                                                       */
/* ------------------------------------------------------------ */

static Part gParts[MAX_PARTS];
static Spring gSprings[MAX_SPRINGS];

static SceneSnapshot gSnapshot;

static int gSelected = -1;
static int gObjectCounter = 0;

static bool gRunning = false;

static ToolType gTool = TOOL_SELECT;
static AxisMode gAxis = AXIS_FREE;

static bool gMouseDragging = false;

static Vector2 gDragStartMouse;

static Vector3 gDragStartPosition;
static Vector3 gDragStartRotation;
static Vector3 gDragStartSize;

static float gDragStartWheel = 0.0f;

static float gPhysicsAccumulator = 0.0f;

static bool gShowExplorer = true;
static bool gShowProperties = true;

static Font gUIFont;
static Font gUIBoldFont;

static bool gUIFontCustom = false;
static bool gUIBoldFontCustom = false;

static Shader gPartShader;

static int gShaderLightLoc = -1;
static int gShaderCameraLoc = -1;
static int gShaderFogLoc = -1;

/* ------------------------------------------------------------ */
/* Names                                                         */
/* ------------------------------------------------------------ */

static const char *PartTypeName(PartType type)
{
    switch (type)
    {
        case PART_BLOCK: return "Block";
        case PART_BALL: return "Ball";
        case PART_CYLINDER: return "Cylinder";
        case PART_BASE: return "Base";
    }

    return "Part";
}

static const char *ToolName(ToolType type)
{
    switch (type)
    {
        case TOOL_SELECT: return "Select";
        case TOOL_MOVE: return "Move";
        case TOOL_ROTATE: return "Rotate";
        case TOOL_SCALE: return "Scale";
        case TOOL_FORCE: return "Force";
    }

    return "Tool";
}

static const char *AxisName(AxisMode axis)
{
    switch (axis)
    {
        case AXIS_X: return "X";
        case AXIS_Y: return "Y";
        case AXIS_Z: return "Z";
        default: return "FREE";
    }
}

static const char *MotorName(MotorType motor)
{
    switch (motor)
    {
        case MOTOR_SPIN: return "Spin";
        case MOTOR_PUSH: return "Push";
        case MOTOR_CONVEYOR: return "Conveyor";
        default: return "None";
    }
}

/* ------------------------------------------------------------ */
/* Model creation                                                 */
/* ------------------------------------------------------------ */

static Model CreateModelForType(PartType type)
{
    Mesh mesh;

    switch (type)
    {
        case PART_BLOCK:
            mesh = GenMeshCube(1, 1, 1);
            break;

        case PART_BALL:
            mesh = GenMeshSphere(0.5f, 32, 20);
            break;

        case PART_CYLINDER:
            mesh = GenMeshCylinder(0.5f, 1.0f, 32);
            break;

        case PART_BASE:
        default:
            mesh = GenMeshCube(1, 1, 1);
            break;
    }

    return LoadModelFromMesh(mesh);
}

static void DestroyPart(Part *p)
{
    if (!p->used)
        return;

    UnloadModel(p->model);

    memset(p, 0, sizeof(Part));
}

/* ------------------------------------------------------------ */
/* Scene                                                         */
/* ------------------------------------------------------------ */

static int FindFreePart(void)
{
    for (int i = 0; i < MAX_PARTS; i++)
    {
        if (!gParts[i].used)
            return i;
    }

    return -1;
}

static void GenerateName(
    Part *p,
    PartType type
)
{
    gObjectCounter++;

    snprintf(
        p->name,
        sizeof(p->name),
        "%s%d",
        PartTypeName(type),
        gObjectCounter
    );
}

static int CreatePart(
    PartType type,
    Vector3 position,
    Vector3 size,
    Color color,
    bool anchored
)
{
    int index = FindFreePart();

    if (index < 0)
        return -1;

    Part *p = &gParts[index];

    memset(p, 0, sizeof(Part));

    p->used = true;
    p->type = type;

    GenerateName(
        p,
        type
    );

    p->position = position;
    p->rotation = V3(0, 0, 0);
    p->size = size;

    p->velocity = V3(0, 0, 0);
    p->angularVelocity = V3(0, 0, 0);

    p->mass =
        type == PART_BASE
        ? 100000.0f
        : 1.0f;

    p->friction = 0.72f;
    p->bounce = 0.22f;

    p->anchored = anchored;
    p->frozen = false;
    p->collisions = true;

    p->sleeping = false;
    p->sleepTimer = 0;

    p->motor = MOTOR_NONE;
    p->motorPower = 8.0f;

    p->hasSpring = false;
    p->springTarget = -1;
    p->springStrength = 20.0f;
    p->springLength = 5.0f;

    p->color = color;

    p->model =
        CreateModelForType(type);

    return index;
}

static void ClearSprings(void)
{
    memset(
        gSprings,
        0,
        sizeof(gSprings)
    );
}

static void CreateInitialScene(void)
{
    for (int i = 0; i < MAX_PARTS; i++)
    {
        if (gParts[i].used)
            DestroyPart(&gParts[i]);
    }

    memset(
        gParts,
        0,
        sizeof(gParts)
    );

    ClearSprings();

    gSelected = -1;
    gObjectCounter = 0;

    /*
        Only the world floor exists initially.
    */
    CreatePart(
        PART_BASE,
        V3(0, -0.5f, 0),
        V3(100, 1, 100),
        (Color){ 158, 163, 168, 255 },
        true
    );
}

static void ResetEditorScene(void)
{
    gRunning = false;
    gSnapshot.valid = false;
    gPhysicsAccumulator = 0;

    CreateInitialScene();
}

/* ------------------------------------------------------------ */
/* Snapshot                                                       */
/* ------------------------------------------------------------ */

static void SaveSimulationSnapshot(void)
{
    memset(
        &gSnapshot,
        0,
        sizeof(gSnapshot)
    );

    gSnapshot.valid = true;
    gSnapshot.selectedPart = gSelected;
    gSnapshot.objectCounter = gObjectCounter;

    for (int i = 0; i < MAX_PARTS; i++)
    {
        if (!gParts[i].used)
            continue;

        gSnapshot.parts[i] =
            gParts[i];

        gSnapshot.parts[i].model =
            (Model){ 0 };
    }

    for (int i = 0; i < MAX_SPRINGS; i++)
    {
        gSnapshot.springs[i] =
            gSprings[i];
    }
}

static void RestoreSimulationSnapshot(void)
{
    if (!gSnapshot.valid)
        return;

    for (int i = 0; i < MAX_PARTS; i++)
    {
        if (gParts[i].used)
            DestroyPart(&gParts[i]);
    }

    memset(
        gParts,
        0,
        sizeof(gParts)
    );

    gObjectCounter =
        gSnapshot.objectCounter;

    for (int i = 0; i < MAX_PARTS; i++)
    {
        if (!gSnapshot.parts[i].used)
            continue;

        gParts[i] =
            gSnapshot.parts[i];

        gParts[i].model =
            CreateModelForType(
                gParts[i].type
            );
    }

    for (int i = 0; i < MAX_SPRINGS; i++)
    {
        gSprings[i] =
            gSnapshot.springs[i];
    }

    gSelected =
        gSnapshot.selectedPart;

    gSnapshot.valid = false;

    gPhysicsAccumulator = 0;
}

static void StartSimulation(void)
{
    if (gRunning)
        return;

    SaveSimulationSnapshot();

    for (int i = 0; i < MAX_PARTS; i++)
    {
        if (!gParts[i].used)
            continue;

        gParts[i].sleeping = false;
        gParts[i].sleepTimer = 0;
    }

    gPhysicsAccumulator = 0;
    gRunning = true;
}

static void StopSimulation(void)
{
    if (!gRunning)
        return;

    gRunning = false;

    RestoreSimulationSnapshot();
}

/* ------------------------------------------------------------ */
/* OBB construction                                               */
/* ------------------------------------------------------------ */

static Matrix PartRotationMatrix(
    const Part *p
)
{
    return MatrixRotateXYZ(
        p->rotation
    );
}

static OBB GetPartOBB(
    const Part *p
)
{
    OBB box;

    box.center =
        p->position;

    box.half =
        V3(
            fabsf(p->size.x) * 0.5f,
            fabsf(p->size.y) * 0.5f,
            fabsf(p->size.z) * 0.5f
        );

    Matrix rotation =
        PartRotationMatrix(p);

    box.axis[0] =
        SafeNormalize(
            Vector3Transform(
                V3(1, 0, 0),
                rotation
            )
        );

    box.axis[1] =
        SafeNormalize(
            Vector3Transform(
                V3(0, 1, 0),
                rotation
            )
        );

    box.axis[2] =
        SafeNormalize(
            Vector3Transform(
                V3(0, 0, 1),
                rotation
            )
        );

    return box;
}

static float PartRadius(
    const Part *p
)
{
    Vector3 h =
        V3(
            fabsf(p->size.x) * 0.5f,
            fabsf(p->size.y) * 0.5f,
            fabsf(p->size.z) * 0.5f
        );

    return sqrtf(
        h.x * h.x +
        h.y * h.y +
        h.z * h.z
    );
}

/* ------------------------------------------------------------ */
/* OBB projection                                                 */
/* ------------------------------------------------------------ */

static float OBBProjectionRadius(
    const OBB *box,
    Vector3 axis
)
{
    return
        box->half.x *
        fabsf(Vector3DotProduct(
            axis,
            box->axis[0]
        ))
        +
        box->half.y *
        fabsf(Vector3DotProduct(
            axis,
            box->axis[1]
        ))
        +
        box->half.z *
        fabsf(Vector3DotProduct(
            axis,
            box->axis[2]
        ));
}

/*
    Full 15-axis OBB SAT.

    This is the important physics upgrade:
    rotated blocks/walls now collide according to their actual
    orientation rather than an axis-aligned box.
*/
static bool OBBvsOBB(
    const OBB *a,
    const OBB *b,
    Vector3 *outNormal,
    float *outPenetration
)
{
    Vector3 axes[15];

    int axisCount = 0;

    axes[axisCount++] = a->axis[0];
    axes[axisCount++] = a->axis[1];
    axes[axisCount++] = a->axis[2];

    axes[axisCount++] = b->axis[0];
    axes[axisCount++] = b->axis[1];
    axes[axisCount++] = b->axis[2];

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            Vector3 cross =
                Vector3CrossProduct(
                    a->axis[i],
                    b->axis[j]
                );

            float length =
                Vector3Length(cross);

            if (length > 0.0001f)
            {
                axes[axisCount++] =
                    Vector3Scale(
                        cross,
                        1.0f / length
                    );
            }
        }
    }

    Vector3 centerDelta =
        Vector3Subtract(
            b->center,
            a->center
        );

    float smallestPenetration =
        100000000.0f;

    Vector3 bestNormal =
        V3(0, 1, 0);

    for (int i = 0; i < axisCount; i++)
    {
        Vector3 axis =
            SafeNormalize(
                axes[i]
            );

        if (Vector3LengthSqr(axis) < 0.000001f)
            continue;

        float radiusA =
            OBBProjectionRadius(
                a,
                axis
            );

        float radiusB =
            OBBProjectionRadius(
                b,
                axis
            );

        float distance =
            fabsf(
                Vector3DotProduct(
                    centerDelta,
                    axis
                )
            );

        float overlap =
            radiusA +
            radiusB -
            distance;

        if (overlap <= 0.0f)
            return false;

        if (overlap <
            smallestPenetration)
        {
            smallestPenetration =
                overlap;

            bestNormal = axis;

            if (Vector3DotProduct(
                    centerDelta,
                    bestNormal
                ) < 0)
            {
                bestNormal =
                    Vector3Negate(
                        bestNormal
                    );
            }
        }
    }

    *outNormal =
        bestNormal;

    *outPenetration =
        smallestPenetration;

    return true;
}

/* ------------------------------------------------------------ */
/* Sphere / OBB                                                   */
/* ------------------------------------------------------------ */

static bool SphereVsOBB(
    Vector3 sphereCenter,
    float radius,
    const OBB *box,
    Vector3 *outNormal,
    float *outPenetration
)
{
    Vector3 relative =
        Vector3Subtract(
            sphereCenter,
            box->center
        );

    /*
        Convert sphere position into the box's local frame.
    */
    Vector3 local =
        V3(
            Vector3DotProduct(
                relative,
                box->axis[0]
            ),
            Vector3DotProduct(
                relative,
                box->axis[1]
            ),
            Vector3DotProduct(
                relative,
                box->axis[2]
            )
        );

    Vector3 closestLocal =
        V3(
            ClampF(
                local.x,
                -box->half.x,
                box->half.x
            ),
            ClampF(
                local.y,
                -box->half.y,
                box->half.y
            ),
            ClampF(
                local.z,
                -box->half.z,
                box->half.z
            )
        );

    Vector3 deltaLocal =
        Vector3Subtract(
            local,
            closestLocal
        );

    float distance =
        Vector3Length(
            deltaLocal
        );

    if (distance > radius)
        return false;

    if (distance > 0.00001f)
    {
        Vector3 localNormal =
            Vector3Scale(
                deltaLocal,
                -1.0f / distance
            );

        Vector3 worldNormal =
            Vector3Add(
                Vector3Add(
                    Vector3Scale(
                        box->axis[0],
                        localNormal.x
                    ),
                    Vector3Scale(
                        box->axis[1],
                        localNormal.y
                    )
                ),
                Vector3Scale(
                    box->axis[2],
                    localNormal.z
                )
            );

        /*
            We need A -> B normal.
            A = sphere
            B = box
        */
        *outNormal =
            SafeNormalize(
                worldNormal
            );

        *outPenetration =
            radius - distance;

        return true;
    }

    /*
        Sphere center is inside the box.
        Find the nearest face.
    */
    float dx =
        box->half.x -
        fabsf(local.x);

    float dy =
        box->half.y -
        fabsf(local.y);

    float dz =
        box->half.z -
        fabsf(local.z);

    Vector3 localNormal;

    float escapeDistance;

    if (dx <= dy && dx <= dz)
    {
        localNormal =
            V3(
                local.x >= 0 ? -1 : 1,
                0,
                0
            );

        escapeDistance = dx;
    }
    else if (dy <= dz)
    {
        localNormal =
            V3(
                0,
                local.y >= 0 ? -1 : 1,
                0
            );

        escapeDistance = dy;
    }
    else
    {
        localNormal =
            V3(
                0,
                0,
                local.z >= 0 ? -1 : 1
            );

        escapeDistance = dz;
    }

    Vector3 worldNormal =
        Vector3Add(
            Vector3Add(
                Vector3Scale(
                    box->axis[0],
                    localNormal.x
                ),
                Vector3Scale(
                    box->axis[1],
                    localNormal.y
                )
            ),
            Vector3Scale(
                box->axis[2],
                localNormal.z
            )
        );

    *outNormal =
        SafeNormalize(
            worldNormal
        );

    *outPenetration =
        radius +
        escapeDistance;

    return true;
}

/* ------------------------------------------------------------ */
/* Broad phase                                                    */
/* ------------------------------------------------------------ */

static bool BroadPhase(
    const Part *a,
    const Part *b
)
{
    float ra =
        PartRadius(a);

    float rb =
        PartRadius(b);

    Vector3 d =
        Vector3Subtract(
            b->position,
            a->position
        );

    float distanceSqr =
        Vector3LengthSqr(d);

    float range =
        ra + rb + 0.25f;

    return distanceSqr <=
           range * range;
}

/* ------------------------------------------------------------ */
/* Contact point approximation                                    */
/* ------------------------------------------------------------ */

static Vector3 ApproxContactPoint(
    const Part *a,
    const Part *b,
    Vector3 normal
)
{
    /*
        Approximate contact positions along the collision normal.
        This gives moving/rotating walls a meaningful surface point.
    */
    float ra =
        PartRadius(a);

    float rb =
        PartRadius(b);

    Vector3 pa =
        Vector3Add(
            a->position,
            Vector3Scale(
                normal,
                ra
            )
        );

    Vector3 pb =
        Vector3Subtract(
            b->position,
            Vector3Scale(
                normal,
                rb
            )
        );

    return Vector3Scale(
        Vector3Add(pa, pb),
        0.5f
    );
}

/* ------------------------------------------------------------ */
/* Collision solver                                               */
/* ------------------------------------------------------------ */

static void ApplyCollision(
    Part *a,
    Part *b,
    Vector3 normal,
    float penetration
)
{
    bool staticA =
        a->anchored ||
        a->frozen;

    bool staticB =
        b->anchored ||
        b->frozen;

    if (staticA && staticB)
        return;

    float invMassA =
        staticA
        ? 0.0f
        : 1.0f /
          fmaxf(
              a->mass,
              0.001f
          );

    float invMassB =
        staticB
        ? 0.0f
        : 1.0f /
          fmaxf(
              b->mass,
              0.001f
          );

    float totalInvMass =
        invMassA +
        invMassB;

    if (totalInvMass <= 0.0f)
        return;

    /*
        Positional correction.
    */
    const float slop = 0.001f;
    const float percent = 0.90f;

    float correctionAmount =
        fmaxf(
            penetration - slop,
            0.0f
        ) *
        percent /
        totalInvMass;

    Vector3 correction =
        Vector3Scale(
            normal,
            correctionAmount
        );

    if (!staticA)
    {
        a->position =
            Vector3Subtract(
                a->position,
                Vector3Scale(
                    correction,
                    invMassA
                )
            );
    }

    if (!staticB)
    {
        b->position =
            Vector3Add(
                b->position,
                Vector3Scale(
                    correction,
                    invMassB
                )
            );
    }

    /*
        Contact velocity.

        A rotating object has velocity at its surface:
            v = linearVelocity + angularVelocity x r

        This is what makes a spinning wall physically interact
        with a ball instead of behaving like a frozen picture.
    */
    Vector3 contact =
        ApproxContactPoint(
            a,
            b,
            normal
        );

    Vector3 ra =
        Vector3Subtract(
            contact,
            a->position
        );

    Vector3 rb =
        Vector3Subtract(
            contact,
            b->position
        );

    Vector3 velocityA =
        a->velocity;

    Vector3 velocityB =
        b->velocity;

    if (!staticA)
    {
        velocityA =
            Vector3Add(
                velocityA,
                Vector3CrossProduct(
                    a->angularVelocity,
                    ra
                )
            );
    }

    if (!staticB)
    {
        velocityB =
            Vector3Add(
                velocityB,
                Vector3CrossProduct(
                    b->angularVelocity,
                    rb
                )
            );
    }

    Vector3 relativeVelocity =
        Vector3Subtract(
            velocityB,
            velocityA
        );

    float normalVelocity =
        Vector3DotProduct(
            relativeVelocity,
            normal
        );

    /*
        Objects are already separating.
    */
    if (normalVelocity > 0.0f)
        return;

    float restitution =
        ClampF(
            fminf(
                a->bounce,
                b->bounce
            ),
            0.0f,
            0.8f
        );

    /*
        Basic rotational effective mass.
        We use approximate inertia values for stable sandbox
        behaviour rather than a fragile full rigid-body tensor.
    */
    float inertiaA =
        staticA
        ? 0.0f
        : fmaxf(
            a->mass *
            PartRadius(a) *
            PartRadius(a) *
            0.4f,
            0.001f
        );

    float inertiaB =
        staticB
        ? 0.0f
        : fmaxf(
            b->mass *
            PartRadius(b) *
            PartRadius(b) *
            0.4f,
            0.001f
        );

    Vector3 raCrossN =
        Vector3CrossProduct(
            ra,
            normal
        );

    Vector3 rbCrossN =
        Vector3CrossProduct(
            rb,
            normal
        );

    float angularMass =
        Vector3LengthSqr(
            raCrossN
        ) /
        fmaxf(
            inertiaA,
            0.001f
        )
        +
        Vector3LengthSqr(
            rbCrossN
        ) /
        fmaxf(
            inertiaB,
            0.001f
        );

    float denominator =
        totalInvMass +
        angularMass;

    float impulseMagnitude =
        -(1.0f + restitution) *
        normalVelocity /
        fmaxf(
            denominator,
            0.0001f
        );

    Vector3 impulse =
        Vector3Scale(
            normal,
            impulseMagnitude
        );

    if (!staticA)
    {
        a->velocity =
            Vector3Subtract(
                a->velocity,
                Vector3Scale(
                    impulse,
                    invMassA
                )
            );

        a->angularVelocity =
            Vector3Subtract(
                a->angularVelocity,
                Vector3Scale(
                    Vector3CrossProduct(
                        ra,
                        impulse
                    ),
                    1.0f /
                    inertiaA
                )
            );
    }

    if (!staticB)
    {
        b->velocity =
            Vector3Add(
                b->velocity,
                Vector3Scale(
                    impulse,
                    invMassB
                )
            );

        b->angularVelocity =
            Vector3Add(
                b->angularVelocity,
                Vector3Scale(
                    Vector3CrossProduct(
                        rb,
                        impulse
                    ),
                    1.0f /
                    inertiaB
                )
            );
    }

    /*
        Friction.
    */
    relativeVelocity =
        Vector3Subtract(
            velocityB,
            velocityA
        );

    Vector3 tangent =
        Vector3Subtract(
            relativeVelocity,
            Vector3Scale(
                normal,
                Vector3DotProduct(
                    relativeVelocity,
                    normal
                )
            )
        );

    float tangentLength =
        Vector3Length(tangent);

    if (tangentLength > 0.0001f)
    {
        tangent =
            Vector3Scale(
                tangent,
                1.0f /
                tangentLength
            );

        float tangentVelocity =
            Vector3DotProduct(
                relativeVelocity,
                tangent
            );

        float jt =
            -tangentVelocity /
            fmaxf(
                totalInvMass,
                0.0001f
            );

        float friction =
            sqrtf(
                ClampF(
                    a->friction,
                    0,
                    1
                )
                *
                ClampF(
                    b->friction,
                    0,
                    1
                )
            );

        float maxFriction =
            impulseMagnitude *
            friction;

        jt =
            ClampF(
                jt,
                -maxFriction,
                maxFriction
            );

        Vector3 frictionImpulse =
            Vector3Scale(
                tangent,
                jt
            );

        if (!staticA)
        {
            a->velocity =
                Vector3Subtract(
                    a->velocity,
                    Vector3Scale(
                        frictionImpulse,
                        invMassA
                    )
                );

            a->angularVelocity =
                Vector3Subtract(
                    a->angularVelocity,
                    Vector3Scale(
                        Vector3CrossProduct(
                            ra,
                            frictionImpulse
                        ),
                        1.0f /
                        inertiaA
                    )
                );
        }

        if (!staticB)
        {
            b->velocity =
                Vector3Add(
                    b->velocity,
                    Vector3Scale(
                        frictionImpulse,
                        invMassB
                    )
                );

            b->angularVelocity =
                Vector3Add(
                    b->angularVelocity,
                    Vector3Scale(
                        Vector3CrossProduct(
                            rb,
                            frictionImpulse
                        ),
                        1.0f /
                        inertiaB
                    )
                );
        }
    }

    a->sleeping = false;
    b->sleeping = false;

    a->sleepTimer = 0;
    b->sleepTimer = 0;
}

/* ------------------------------------------------------------ */
/* Pair collision                                                 */
/* ------------------------------------------------------------ */

static void ResolvePair(
    Part *a,
    Part *b
)
{
    Vector3 normal;
    float penetration;

    bool aBall =
        a->type == PART_BALL;

    bool bBall =
        b->type == PART_BALL;

    if (aBall && bBall)
    {
        Vector3 delta =
            Vector3Subtract(
                b->position,
                a->position
            );

        float distance =
            Vector3Length(delta);

        float radiusA =
            a->size.x * 0.5f;

        float radiusB =
            b->size.x * 0.5f;

        float combined =
            radiusA +
            radiusB;

        if (distance >= combined)
            return;

        if (distance < 0.00001f)
        {
            normal =
                V3(0, 1, 0);

            penetration =
                combined;
        }
        else
        {
            normal =
                Vector3Scale(
                    delta,
                    1.0f /
                    distance
                );

            penetration =
                combined -
                distance;
        }

        ApplyCollision(
            a,
            b,
            normal,
            penetration
        );

        return;
    }

    if (aBall && !bBall)
    {
        OBB box =
            GetPartOBB(b);

        if (SphereVsOBB(
                a->position,
                a->size.x * 0.5f,
                &box,
                &normal,
                &penetration))
        {
            ApplyCollision(
                a,
                b,
                normal,
                penetration
            );
        }

        return;
    }

    if (!aBall && bBall)
    {
        OBB box =
            GetPartOBB(a);

        if (SphereVsOBB(
                b->position,
                b->size.x * 0.5f,
                &box,
                &normal,
                &penetration))
        {
            /*
                SphereVsOBB gives sphere -> box.
                We need A -> B.
            */
            normal =
                Vector3Negate(
                    normal
                );

            ApplyCollision(
                a,
                b,
                normal,
                penetration
            );
        }

        return;
    }

    OBB boxA =
        GetPartOBB(a);

    OBB boxB =
        GetPartOBB(b);

    if (OBBvsOBB(
            &boxA,
            &boxB,
            &normal,
            &penetration))
    {
        ApplyCollision(
            a,
            b,
            normal,
            penetration
        );
    }
}

/* ------------------------------------------------------------ */
/* Ground collision                                               */
/* ------------------------------------------------------------ */

static void ResolveGround(
    Part *p
)
{
    if (!p->used ||
        p->type == PART_BASE)
        return;

    if (p->anchored ||
        p->frozen ||
        !p->collisions)
        return;

    /*
        Use the actual rotated OBB to determine the lowest point.
    */
    OBB box =
        GetPartOBB(p);

    float bottom =
        box.center.y -
        (
            fabsf(box.axis[0].y) *
            box.half.x
            +
            fabsf(box.axis[1].y) *
            box.half.y
            +
            fabsf(box.axis[2].y) *
            box.half.z
        );

    if (bottom >= 0)
        return;

    float penetration =
        -bottom;

    p->position.y +=
        penetration;

    Vector3 normal =
        V3(0, 1, 0);

    Vector3 contact =
        Vector3Add(
            p->position,
            V3(
                0,
                -(
                    box.half.y
                ),
                0
            )
        );

    Vector3 r =
        Vector3Subtract(
            contact,
            p->position
        );

    Vector3 contactVelocity =
        Vector3Add(
            p->velocity,
            Vector3CrossProduct(
                p->angularVelocity,
                r
            )
        );

    float normalVelocity =
        Vector3DotProduct(
            contactVelocity,
            normal
        );

    if (normalVelocity < 0)
    {
        float invMass =
            1.0f /
            fmaxf(
                p->mass,
                0.001f
            );

        float inertia =
            p->mass *
            PartRadius(p) *
            PartRadius(p) *
            0.4f;

        Vector3 rCrossN =
            Vector3CrossProduct(
                r,
                normal
            );

        float denominator =
            invMass +
            Vector3LengthSqr(
                rCrossN
            ) /
            fmaxf(
                inertia,
                0.001f
            );

        float restitution =
            ClampF(
                p->bounce,
                0,
                0.8f
            );

        float j =
            -(1.0f + restitution) *
            normalVelocity /
            fmaxf(
                denominator,
                0.0001f
            );

        Vector3 impulse =
            Vector3Scale(
                normal,
                j
            );

        p->velocity =
            Vector3Add(
                p->velocity,
                Vector3Scale(
                    impulse,
                    invMass
                )
            );

        p->angularVelocity =
            Vector3Add(
                p->angularVelocity,
                Vector3Scale(
                    Vector3CrossProduct(
                        r,
                        impulse
                    ),
                    1.0f /
                    fmaxf(
                        inertia,
                        0.001f
                    )
                )
            );
    }

    /*
        Ground friction.
    */
    float friction =
        ClampF(
            p->friction,
            0,
            1
        );

    p->velocity.x *=
        1.0f -
        friction * 0.08f;

    p->velocity.z *=
        1.0f -
        friction * 0.08f;

    p->angularVelocity.x *=
        0.985f;

    p->angularVelocity.z *=
        0.985f;

    p->sleeping = false;
    p->sleepTimer = 0;
}

/* ------------------------------------------------------------ */
/* Sleeping                                                       */
/* ------------------------------------------------------------ */

static void UpdateSleeping(
    Part *p,
    float dt
)
{
    if (p->anchored ||
        p->frozen)
        return;

    float speed =
        Vector3LengthSqr(
            p->velocity
        );

    float angular =
        Vector3LengthSqr(
            p->angularVelocity
        );

    if (speed < 0.004f &&
        angular < 0.004f)
    {
        p->sleepTimer += dt;

        if (p->sleepTimer >
            0.8f)
        {
            p->sleeping = true;

            p->velocity =
                V3(0, 0, 0);

            p->angularVelocity =
                V3(0, 0, 0);
        }
    }
    else
    {
        p->sleepTimer = 0;
        p->sleeping = false;
    }
}

/* ------------------------------------------------------------ */
/* Motors                                                        */
/* ------------------------------------------------------------ */

static void ApplyMotor(
    Part *p,
    float dt
)
{
    if (p->motor == MOTOR_NONE)
        return;

    if (p->anchored ||
        p->frozen)
        return;

    p->sleeping = false;
    p->sleepTimer = 0;

    if (p->motor == MOTOR_SPIN)
    {
        /*
            A strong but controllable angular motor.
        */
        float target =
            p->motorPower;

        float difference =
            target -
            p->angularVelocity.y;

        p->angularVelocity.y +=
            difference *
            ClampF(
                dt * 12.0f,
                0,
                1
            );
    }
    else if (p->motor == MOTOR_PUSH)
    {
        Vector3 forward =
            Vector3Transform(
                V3(0, 0, 1),
                PartRotationMatrix(p)
            );

        forward =
            SafeNormalize(forward);

        p->velocity =
            Vector3Add(
                p->velocity,
                Vector3Scale(
                    forward,
                    p->motorPower *
                    dt
                )
            );
    }
    else if (p->motor == MOTOR_CONVEYOR)
    {
        Vector3 forward =
            Vector3Transform(
                V3(1, 0, 0),
                PartRotationMatrix(p)
            );

        forward.y = 0;

        forward =
            SafeNormalize(forward);

        p->velocity =
            Vector3Add(
                p->velocity,
                Vector3Scale(
                    forward,
                    p->motorPower *
                    dt
                )
            );
    }
}

/* ------------------------------------------------------------ */
/* Springs                                                        */
/* ------------------------------------------------------------ */

static void ApplySprings(
    float dt
)
{
    for (int i = 0;
         i < MAX_SPRINGS;
         i++)
    {
        Spring *s =
            &gSprings[i];

        if (!s->used)
            continue;

        if (s->a < 0 ||
            s->a >= MAX_PARTS ||
            s->b < 0 ||
            s->b >= MAX_PARTS)
            continue;

        Part *a =
            &gParts[s->a];

        Part *b =
            &gParts[s->b];

        if (!a->used ||
            !b->used)
            continue;

        Vector3 delta =
            Vector3Subtract(
                b->position,
                a->position
            );

        float length =
            Vector3Length(delta);

        if (length < 0.0001f)
            continue;

        Vector3 direction =
            Vector3Scale(
                delta,
                1.0f / length
            );

        float extension =
            length -
            s->restLength;

        Vector3 relativeVelocity =
            Vector3Subtract(
                b->velocity,
                a->velocity
            );

        float velocityAlong =
            Vector3DotProduct(
                relativeVelocity,
                direction
            );

        float force =
            extension *
            s->stiffness -
            velocityAlong *
            s->damping;

        Vector3 impulseForce =
            Vector3Scale(
                direction,
                force
            );

        if (!a->anchored &&
            !a->frozen)
        {
            a->velocity =
                Vector3Add(
                    a->velocity,
                    Vector3Scale(
                        impulseForce,
                        dt /
                        fmaxf(
                            a->mass,
                            0.01f
                        )
                    )
                );
        }

        if (!b->anchored &&
            !b->frozen)
        {
            b->velocity =
                Vector3Subtract(
                    b->velocity,
                    Vector3Scale(
                        impulseForce,
                        dt /
                        fmaxf(
                            b->mass,
                            0.01f
                        )
                    )
                );
        }
    }
}

/* ------------------------------------------------------------ */
/* Physics step                                                   */
/* ------------------------------------------------------------ */

static void PhysicsStep(
    float dt
)
{
    ApplySprings(dt);

    /*
        Integrate.
    */
    for (int i = 0;
         i < MAX_PARTS;
         i++)
    {
        Part *p =
            &gParts[i];

        if (!p->used ||
            p->type == PART_BASE)
            continue;

        ApplyMotor(
            p,
            dt
        );

        if (p->anchored ||
            p->frozen ||
            p->sleeping)
            continue;

        p->velocity.y +=
            GRAVITY *
            dt;

        /*
            Light aerodynamic damping.
        */
        float damping =
            powf(
                0.999f,
                dt * 240.0f
            );

        p->velocity =
            Vector3Scale(
                p->velocity,
                damping
            );

        p->angularVelocity =
            Vector3Scale(
                p->angularVelocity,
                powf(
                    0.9995f,
                    dt * 240.0f
                )
            );

        /*
            Hard velocity safety.
        */
        float speed =
            Vector3Length(
                p->velocity
            );

        if (speed >
            MAX_VELOCITY)
        {
            p->velocity =
                Vector3Scale(
                    p->velocity,
                    MAX_VELOCITY /
                    speed
                );
        }

        float angularSpeed =
            Vector3Length(
                p->angularVelocity
            );

        if (angularSpeed >
            MAX_ANGULAR_VELOCITY)
        {
            p->angularVelocity =
                Vector3Scale(
                    p->angularVelocity,
                    MAX_ANGULAR_VELOCITY /
                    angularSpeed
                );
        }

        p->position =
            Vector3Add(
                p->position,
                Vector3Scale(
                    p->velocity,
                    dt
                )
            );

        p->rotation =
            Vector3Add(
                p->rotation,
                Vector3Scale(
                    p->angularVelocity,
                    dt
                )
            );
    }

    /*
        Solver.

        Ground is solved repeatedly because a moving wall can
        push an object into the floor during the same timestep.
    */
    for (int iteration = 0;
         iteration < SOLVER_ITERATIONS;
         iteration++)
    {
        for (int i = 0;
             i < MAX_PARTS;
             i++)
        {
            if (!gParts[i].used)
                continue;

            ResolveGround(
                &gParts[i]
            );
        }

        for (int i = 0;
             i < MAX_PARTS;
             i++)
        {
            Part *a =
                &gParts[i];

            if (!a->used ||
                !a->collisions ||
                a->type == PART_BASE)
                continue;

            for (int j = i + 1;
                 j < MAX_PARTS;
                 j++)
            {
                Part *b =
                    &gParts[j];

                if (!b->used ||
                    !b->collisions ||
                    b->type == PART_BASE)
                    continue;

                if (!BroadPhase(
                        a,
                        b))
                    continue;

                ResolvePair(
                    a,
                    b
                );
            }
        }
    }

    for (int i = 0;
         i < MAX_PARTS;
         i++)
    {
        if (!gParts[i].used)
            continue;

        UpdateSleeping(
            &gParts[i],
            dt
        );
    }
}

/* ------------------------------------------------------------ */
/* Picking                                                        */
/* ------------------------------------------------------------ */

static bool RaySphereHit(
    Ray ray,
    Vector3 center,
    float radius,
    float *distance
)
{
    Vector3 oc =
        Vector3Subtract(
            ray.position,
            center
        );

    float a =
        Vector3DotProduct(
            ray.direction,
            ray.direction
        );

    float b =
        2.0f *
        Vector3DotProduct(
            oc,
            ray.direction
        );

    float c =
        Vector3DotProduct(
            oc,
            oc
        ) -
        radius * radius;

    float discriminant =
        b * b -
        4.0f * a * c;

    if (discriminant < 0)
        return false;

    float root =
        sqrtf(discriminant);

    float t =
        (-b - root) /
        (2.0f * a);

    if (t < 0)
    {
        t =
            (-b + root) /
            (2.0f * a);
    }

    if (t < 0)
        return false;

    if (distance)
        *distance = t;

    return true;
}

static int PickPart(
    Ray ray
)
{
    int selected = -1;

    float closest =
        100000000.0f;

    for (int i = 0;
         i < MAX_PARTS;
         i++)
    {
        Part *p =
            &gParts[i];

        if (!p->used)
            continue;

        if (p->type == PART_BASE)
            continue;

        float distance;

        if (RaySphereHit(
                ray,
                p->position,
                PartRadius(p),
                &distance))
        {
            if (distance <
                closest)
            {
                closest =
                    distance;

                selected =
                    i;
            }
        }
    }

    return selected;
}

/* ------------------------------------------------------------ */
/* Camera                                                        */
/* ------------------------------------------------------------ */

static void UpdateEditorCamera(
    EditorCamera *ec
)
{
    ec->pitch =
        ClampF(
            ec->pitch,
            -1.45f,
            1.45f
        );

    ec->distance =
        ClampF(
            ec->distance,
            2.5f,
            250.0f
        );

    float cp =
        cosf(ec->pitch);

    ec->camera.position =
        V3(
            ec->target.x +
                sinf(ec->yaw) *
                cp *
                ec->distance,

            ec->target.y +
                sinf(ec->pitch) *
                ec->distance,

            ec->target.z +
                cosf(ec->yaw) *
                cp *
                ec->distance
        );

    ec->camera.target =
        ec->target;

    ec->camera.up =
        V3(0, 1, 0);
}

static void UpdateCameraInput(
    EditorCamera *ec
)
{
    if (IsMouseButtonPressed(
            MOUSE_BUTTON_RIGHT))
        ec->orbit = true;

    if (IsMouseButtonReleased(
            MOUSE_BUTTON_RIGHT))
        ec->orbit = false;

    if (IsMouseButtonPressed(
            MOUSE_BUTTON_MIDDLE))
        ec->pan = true;

    if (IsMouseButtonReleased(
            MOUSE_BUTTON_MIDDLE))
        ec->pan = false;

    if (ec->orbit)
    {
        Vector2 delta =
            GetMouseDelta();

        ec->yaw -=
            delta.x * 0.008f;

        ec->pitch -=
            delta.y * 0.008f;
    }

    if (ec->pan)
    {
        Vector2 delta =
            GetMouseDelta();

        Vector3 forward =
            Vector3Subtract(
                ec->camera.target,
                ec->camera.position
            );

        forward.y = 0;

        forward =
            SafeNormalize(
                forward
            );

        Vector3 right =
            V3(
                forward.z,
                0,
                -forward.x
            );

        float panSpeed =
            ec->distance *
            0.0018f;

        ec->target =
            Vector3Add(
                ec->target,
                Vector3Scale(
                    right,
                    -delta.x *
                    panSpeed
                )
            );

        ec->target =
            Vector3Add(
                ec->target,
                Vector3Scale(
                    forward,
                    delta.y *
                    panSpeed
                )
            );
    }

    float wheel =
        GetMouseWheelMove();

    if (fabsf(wheel) > 0.001f)
    {
        ec->distance -=
            wheel *
            fmaxf(
                0.8f,
                ec->distance *
                0.08f
            );
    }

    /*
        WASD is now ALWAYS camera movement.

        This was intentionally separated from the Move tool.
        Previously W selected Move and camera movement was only
        available while orbiting/panning, which made WASD feel
        broken.
    */
    Vector3 forward =
        Vector3Subtract(
            ec->camera.target,
            ec->camera.position
        );

    forward.y = 0;

    forward =
        SafeNormalize(
            forward
        );

    Vector3 right =
        V3(
            forward.z,
            0,
            -forward.x
        );

    float speed =
        IsKeyDown(KEY_LEFT_SHIFT)
        ? 1.0f
        : 0.28f;

    if (IsKeyDown(KEY_W))
    {
        ec->target =
            Vector3Add(
                ec->target,
                Vector3Scale(
                    forward,
                    speed
                )
            );
    }

    if (IsKeyDown(KEY_S))
    {
        ec->target =
            Vector3Subtract(
                ec->target,
                Vector3Scale(
                    forward,
                    speed
                )
            );
    }

    if (IsKeyDown(KEY_D))
    {
        ec->target =
            Vector3Add(
                ec->target,
                Vector3Scale(
                    right,
                    speed
                )
            );
    }

    if (IsKeyDown(KEY_A))
    {
        ec->target =
            Vector3Subtract(
                ec->target,
                Vector3Scale(
                    right,
                    speed
                )
            );
    }

    if (IsKeyDown(KEY_Q) &&
        IsKeyDown(KEY_LEFT_ALT))
    {
        ec->target.y += speed;
    }

    if (IsKeyDown(KEY_E) &&
        IsKeyDown(KEY_LEFT_ALT))
    {
        ec->target.y -= speed;
    }

    UpdateEditorCamera(ec);
}

/* ------------------------------------------------------------ */
/* Editor mouse                                                   */
/* ------------------------------------------------------------ */

static void UpdateEditorMouse(
    EditorCamera *ec
)
{
    Vector2 mouse =
        GetMousePosition();

    /*
        UI exclusion.
    */
    if (mouse.y < 100)
        return;

    Ray ray =
        GetScreenToWorldRay(
            mouse,
            ec->camera
        );

    if (IsMouseButtonPressed(
            MOUSE_BUTTON_LEFT))
    {
        int hit =
            PickPart(ray);

        if (hit < 0)
        {
            if (gTool == TOOL_SELECT)
                gSelected = -1;

            return;
        }

        gSelected = hit;

        Part *p =
            &gParts[gSelected];

        gDragStartMouse =
            mouse;

        gDragStartPosition =
            p->position;

        gDragStartRotation =
            p->rotation;

        gDragStartSize =
            p->size;

        gDragStartWheel =
            GetMouseWheelMove();

        if (gTool != TOOL_SELECT)
        {
            gMouseDragging = true;
        }
    }

    if (!gMouseDragging ||
        gSelected < 0 ||
        !gParts[gSelected].used)
        return;

    if (!IsMouseButtonDown(
            MOUSE_BUTTON_LEFT))
    {
        gMouseDragging = false;
        return;
    }

    Part *p =
        &gParts[gSelected];

    if (p->anchored ||
        p->frozen)
        return;

    Vector2 delta =
        Vector2Subtract(
            mouse,
            gDragStartMouse
        );

    /*
        --------------------------------------------------------
        MOVE
        --------------------------------------------------------
    */
    if (gTool == TOOL_MOVE)
    {
        float speed =
            ec->distance *
            0.0024f;

        if (gAxis == AXIS_Y)
        {
            p->position.y =
                gDragStartPosition.y -
                delta.y *
                speed;
        }
        else if (gAxis == AXIS_X)
        {
            p->position.x =
                gDragStartPosition.x +
                delta.x *
                speed;
        }
        else if (gAxis == AXIS_Z)
        {
            p->position.z =
                gDragStartPosition.z -
                delta.x *
                speed;
        }
        else
        {
            Vector3 forward =
                Vector3Subtract(
                    ec->camera.target,
                    ec->camera.position
                );

            forward.y = 0;

            forward =
                SafeNormalize(
                    forward
                );

            Vector3 right =
                V3(
                    forward.z,
                    0,
                    -forward.x
                );

            Vector3 movement =
                Vector3Add(
                    Vector3Scale(
                        right,
                        delta.x *
                        speed
                    ),
                    Vector3Scale(
                        forward,
                        -delta.y *
                        speed
                    )
                );

            p->position =
                Vector3Add(
                    gDragStartPosition,
                    movement
                );
        }

        /*
            Mouse wheel = vertical lift while moving.
        */
        float wheel =
            GetMouseWheelMove();

        if (fabsf(wheel) > 0.001f)
        {
            p->position.y +=
                wheel *
                0.5f;
        }

        p->velocity =
            V3(0, 0, 0);

        p->sleeping = false;
        p->sleepTimer = 0;
    }

    /*
        --------------------------------------------------------
        ROTATE
        --------------------------------------------------------
    */
    if (gTool == TOOL_ROTATE)
    {
        if (gAxis == AXIS_X)
        {
            p->rotation.x =
                gDragStartRotation.x +
                delta.x * 0.012f;
        }
        else if (gAxis == AXIS_Y)
        {
            p->rotation.y =
                gDragStartRotation.y +
                delta.x * 0.012f;
        }
        else if (gAxis == AXIS_Z)
        {
            p->rotation.z =
                gDragStartRotation.z +
                delta.x * 0.012f;
        }
        else
        {
            p->rotation.y =
                gDragStartRotation.y +
                delta.x * 0.012f;

            p->rotation.x =
                gDragStartRotation.x -
                delta.y * 0.012f;
        }
    }

    /*
        --------------------------------------------------------
        SCALE
        --------------------------------------------------------
    */
    if (gTool == TOOL_SCALE)
    {
        if (IsKeyDown(KEY_LEFT_CONTROL))
        {
            float amount =
                1.0f +
                delta.x * 0.012f;

            amount =
                ClampF(
                    amount,
                    0.05f,
                    20.0f
                );

            p->size =
                Vector3Scale(
                    gDragStartSize,
                    amount
                );
        }
        else if (gAxis == AXIS_X)
        {
            float amount =
                1.0f +
                delta.x * 0.012f;

            p->size.x =
                fmaxf(
                    0.1f,
                    gDragStartSize.x *
                    ClampF(
                        amount,
                        0.05f,
                        20.0f
                    )
                );
        }
        else if (gAxis == AXIS_Y)
        {
            float amount =
                1.0f -
                delta.y * 0.012f;

            p->size.y =
                fmaxf(
                    0.1f,
                    gDragStartSize.y *
                    ClampF(
                        amount,
                        0.05f,
                        20.0f
                    )
                );
        }
        else if (gAxis == AXIS_Z)
        {
            float amount =
                1.0f -
                delta.y * 0.012f;

            p->size.z =
                fmaxf(
                    0.1f,
                    gDragStartSize.z *
                    ClampF(
                        amount,
                        0.05f,
                        20.0f
                    )
                );
        }
        else if (IsKeyDown(KEY_LEFT_SHIFT))
        {
            float amount =
                1.0f -
                delta.y * 0.012f;

            p->size.y =
                fmaxf(
                    0.1f,
                    gDragStartSize.y *
                    ClampF(
                        amount,
                        0.05f,
                        20.0f
                    )
                );
        }
        else
        {
            float xAmount =
                1.0f +
                delta.x * 0.012f;

            float zAmount =
                1.0f -
                delta.y * 0.012f;

            p->size.x =
                fmaxf(
                    0.1f,
                    gDragStartSize.x *
                    ClampF(
                        xAmount,
                        0.05f,
                        20.0f
                    )
                );

            p->size.z =
                fmaxf(
                    0.1f,
                    gDragStartSize.z *
                    ClampF(
                        zAmount,
                        0.05f,
                        20.0f
                    )
                );
        }
    }

    /*
        --------------------------------------------------------
        FORCE
        --------------------------------------------------------
    */
    if (gTool == TOOL_FORCE)
    {
        Vector3 forward =
            SafeNormalize(
                Vector3Subtract(
                    ec->camera.target,
                    ec->camera.position
                )
            );

        float strength =
            (-delta.y +
             delta.x * 0.25f) *
            0.045f;

        p->velocity =
            Vector3Add(
                p->velocity,
                Vector3Scale(
                    forward,
                    strength
                )
            );

        p->sleeping = false;
        p->sleepTimer = 0;
    }
}

/* ------------------------------------------------------------ */
/* Keyboard tool operations                                       */
/* ------------------------------------------------------------ */

static void UpdateKeyboardTools(void)
{
    if (gRunning)
        return;

    if (gSelected < 0 ||
        !gParts[gSelected].used)
        return;

    Part *p =
        &gParts[gSelected];

    if (p->anchored ||
        p->frozen)
        return;

    float movement =
        IsKeyDown(KEY_LEFT_SHIFT)
        ? 0.20f
        : 0.045f;

    if (gTool == TOOL_MOVE)
    {
        if (IsKeyDown(KEY_LEFT))
            p->position.x -= movement;

        if (IsKeyDown(KEY_RIGHT))
            p->position.x += movement;

        if (IsKeyDown(KEY_UP))
            p->position.z -= movement;

        if (IsKeyDown(KEY_DOWN))
            p->position.z += movement;

        /*
            PageUp/PageDown AND keypad +/- provide vertical
            movement so this is much harder to accidentally
            get stuck without a Y control.
        */
        if (IsKeyDown(KEY_PAGE_UP) ||
            IsKeyDown(KEY_KP_ADD))
        {
            p->position.y += movement;
        }

        if (IsKeyDown(KEY_PAGE_DOWN) ||
            IsKeyDown(KEY_KP_SUBTRACT))
        {
            p->position.y -= movement;
        }
    }

    if (gTool == TOOL_ROTATE)
    {
        if (IsKeyDown(KEY_LEFT))
            p->rotation.y -= 0.02f;

        if (IsKeyDown(KEY_RIGHT))
            p->rotation.y += 0.02f;

        if (IsKeyDown(KEY_UP))
            p->rotation.x -= 0.02f;

        if (IsKeyDown(KEY_DOWN))
            p->rotation.x += 0.02f;

        if (IsKeyDown(KEY_PAGE_UP))
            p->rotation.z += 0.02f;

        if (IsKeyDown(KEY_PAGE_DOWN))
            p->rotation.z -= 0.02f;
    }

    if (gTool == TOOL_SCALE)
    {
        float amount =
            IsKeyDown(KEY_LEFT_SHIFT)
            ? 0.10f
            : 0.04f;

        if (IsKeyDown(KEY_EQUAL))
        {
            if (gAxis == AXIS_X)
                p->size.x += amount;
            else if (gAxis == AXIS_Y)
                p->size.y += amount;
            else if (gAxis == AXIS_Z)
                p->size.z += amount;
            else
            {
                p->size.x += amount;
                p->size.z += amount;
            }
        }

        if (IsKeyDown(KEY_MINUS))
        {
            if (gAxis == AXIS_X)
                p->size.x =
                    fmaxf(
                        0.1f,
                        p->size.x -
                        amount
                    );
            else if (gAxis == AXIS_Y)
                p->size.y =
                    fmaxf(
                        0.1f,
                        p->size.y -
                        amount
                    );
            else if (gAxis == AXIS_Z)
                p->size.z =
                    fmaxf(
                        0.1f,
                        p->size.z -
                        amount
                    );
            else
            {
                p->size.x =
                    fmaxf(
                        0.1f,
                        p->size.x -
                        amount
                    );

                p->size.z =
                    fmaxf(
                        0.1f,
                        p->size.z -
                        amount
                    );
            }
        }
    }
}

/* ------------------------------------------------------------ */
/* Object creation                                                */
/* ------------------------------------------------------------ */

static void NewBlock(void)
{
    int index =
        CreatePart(
            PART_BLOCK,
            V3(0, 5, 0),
            V3(4, 4, 4),
            (Color){ 75, 145, 215, 255 },
            false
        );

    if (index >= 0)
        gSelected = index;
}

static void NewBall(void)
{
    int index =
        CreatePart(
            PART_BALL,
            V3(0, 7, 0),
            V3(3, 3, 3),
            (Color){ 215, 90, 80, 255 },
            false
        );

    if (index >= 0)
        gSelected = index;
}

static void NewCylinder(void)
{
    int index =
        CreatePart(
            PART_CYLINDER,
            V3(0, 5, 0),
            V3(3, 5, 3),
            (Color){ 85, 185, 115, 255 },
            false
        );

    if (index >= 0)
        gSelected = index;
}

static void NewBase(void)
{
    int index =
        CreatePart(
            PART_BASE,
            V3(0, 0.1f, 0),
            V3(20, 0.2f, 20),
            (Color){ 155, 160, 165, 255 },
            true
        );

    if (index >= 0)
        gSelected = index;
}

/* ------------------------------------------------------------ */
/* Delete / duplicate                                             */
/* ------------------------------------------------------------ */

static void DeleteSelected(void)
{
    if (gSelected < 0)
        return;

    if (!gParts[gSelected].used)
        return;

    int deleted =
        gSelected;

    DestroyPart(
        &gParts[deleted]
    );

    for (int i = 0;
         i < MAX_SPRINGS;
         i++)
    {
        if (!gSprings[i].used)
            continue;

        if (gSprings[i].a == deleted ||
            gSprings[i].b == deleted)
        {
            gSprings[i].used = false;
        }
    }

    gSelected = -1;
}

static void DuplicateSelected(void)
{
    if (gSelected < 0 ||
        !gParts[gSelected].used)
        return;

    int index =
        FindFreePart();

    if (index < 0)
        return;

    Part *source =
        &gParts[gSelected];

    Part *dest =
        &gParts[index];

    *dest =
        *source;

    dest->used = true;

    gObjectCounter++;

    snprintf(
        dest->name,
        sizeof(dest->name),
        "%s%d",
        PartTypeName(dest->type),
        gObjectCounter
    );

    dest->position =
        Vector3Add(
            source->position,
            V3(3, 2, 3)
        );

    dest->velocity =
        V3(0, 0, 0);

    dest->angularVelocity =
        V3(0, 0, 0);

    dest->sleeping = false;
    dest->sleepTimer = 0;

    dest->hasSpring = false;
    dest->springTarget = -1;

    dest->model =
        CreateModelForType(
            dest->type
        );

    gSelected = index;
}

/* ------------------------------------------------------------ */
/* Shader                                                         */
/* ------------------------------------------------------------ */

static void CreatePartShader(void)
{
    const char *vertexCode =
        "#version 330\n"

        "in vec3 vertexPosition;\n"
        "in vec3 vertexNormal;\n"

        "uniform mat4 mvp;\n"
        "uniform mat4 matModel;\n"

        "out vec3 fragPosition;\n"
        "out vec3 fragNormal;\n"

        "void main()\n"
        "{\n"
        "    vec4 worldPosition = matModel * vec4(vertexPosition, 1.0);\n"
        "    fragPosition = worldPosition.xyz;\n"
        "    fragNormal = normalize(mat3(matModel) * vertexNormal);\n"
        "    gl_Position = mvp * vec4(vertexPosition, 1.0);\n"
        "}\n";

    const char *fragmentCode =
        "#version 330\n"

        "in vec3 fragPosition;\n"
        "in vec3 fragNormal;\n"

        "uniform vec4 colDiffuse;\n"
        "uniform vec3 lightDir;\n"
        "uniform vec3 cameraPosition;\n"
        "uniform vec4 fogColor;\n"
        "uniform float fogDensity;\n"

        "out vec4 finalColor;\n"

        "void main()\n"
        "{\n"
        "    vec3 n = normalize(fragNormal);\n"
        "    vec3 l = normalize(-lightDir);\n"
        "    vec3 v = normalize(cameraPosition - fragPosition);\n"
        "    vec3 h = normalize(l + v);\n"

        "    float diffuse = max(dot(n, l), 0.0);\n"

        "    float specular = pow(\n"
        "        max(dot(n, h), 0.0),\n"
        "        28.0\n"
        "    ) * 0.07;\n"

        "    float light =\n"
        "        0.40 +\n"
        "        diffuse * 0.60;\n"

        "    vec3 color =\n"
        "        colDiffuse.rgb * light +\n"
        "        vec3(specular);\n"

        "    float distanceToCamera =\n"
        "        length(cameraPosition - fragPosition);\n"

        "    float fog = 1.0 - exp(\n"
        "        -fogDensity * distanceToCamera\n"
        "    );\n"

        "    fog = clamp(fog, 0.0, 1.0);\n"

        "    color = mix(\n"
        "        color,\n"
        "        fogColor.rgb,\n"
        "        fog\n"
        "    );\n"

        "    finalColor = vec4(\n"
        "        color,\n"
        "        colDiffuse.a\n"
        "    );\n"
        "}\n";

    gPartShader =
        LoadShaderFromMemory(
            vertexCode,
            fragmentCode
        );

    if (gPartShader.id <= 0)
        return;

    gShaderLightLoc =
        GetShaderLocation(
            gPartShader,
            "lightDir"
        );

    gShaderCameraLoc =
        GetShaderLocation(
            gPartShader,
            "cameraPosition"
        );

    gShaderFogLoc =
        GetShaderLocation(
            gPartShader,
            "fogColor"
        );
}

/* ------------------------------------------------------------ */
/* Rendering transform                                            */
/* ------------------------------------------------------------ */

static Matrix PartTransform(
    const Part *p
)
{
    Matrix scale =
        MatrixScale(
            p->size.x,
            p->size.y,
            p->size.z
        );

    Matrix rotation =
        MatrixRotateXYZ(
            p->rotation
        );

    Matrix translation =
        MatrixTranslate(
            p->position.x,
            p->position.y,
            p->position.z
        );

    /*
        Same general order as raylib's extended model transform:
            scale -> rotation -> translation
    */
    return MatrixMultiply(
        MatrixMultiply(
            scale,
            rotation
        ),
        translation
    );
}

static void DrawPartGeometry(
    const Part *p
)
{
    if (!p->used)
        return;

    Matrix transform =
        PartTransform(p);

    for (int meshIndex = 0;
         meshIndex < p->model.meshCount;
         meshIndex++)
    {
        int materialIndex = 0;

        if (p->model.meshMaterial)
        {
            materialIndex =
                p->model.meshMaterial[
                    meshIndex
                ];
        }

        if (materialIndex < 0 ||
            materialIndex >=
            p->model.materialCount)
        {
            materialIndex = 0;
        }

        Material material =
            p->model.materials[
                materialIndex
            ];

        if (material.maps)
        {
            material.maps[
                MATERIAL_MAP_DIFFUSE
            ].color =
                p->color;
        }

        DrawMesh(
            p->model.meshes[
                meshIndex
            ],
            material,
            transform
        );
    }
}

/* ------------------------------------------------------------ */
/* Selection                                                      */
/* ------------------------------------------------------------ */

static void DrawSelection(
    const Part *p
)
{
    if (!p)
        return;

    float radius =
        PartRadius(p) +
        0.45f;

    DrawCircle3D(
        p->position,
        radius,
        V3(1, 0, 0),
        90,
        (Color){
            45,
            135,
            235,
            220
        }
    );

    if (gTool == TOOL_MOVE)
    {
        float handle =
            radius;

        Vector3 x =
            Vector3Add(
                p->position,
                V3(handle, 0, 0)
            );

        Vector3 y =
            Vector3Add(
                p->position,
                V3(0, handle, 0)
            );

        Vector3 z =
            Vector3Add(
                p->position,
                V3(0, 0, handle)
            );

        DrawLine3D(
            p->position,
            x,
            (Color){ 225, 70, 70, 255 }
        );

        DrawLine3D(
            p->position,
            y,
            (Color){ 70, 200, 90, 255 }
        );

        DrawLine3D(
            p->position,
            z,
            (Color){ 70, 110, 230, 255 }
        );

        DrawSphere(
            x,
            0.13f,
            (Color){ 225, 70, 70, 255 }
        );

        DrawSphere(
            y,
            0.13f,
            (Color){ 70, 200, 90, 255 }
        );

        DrawSphere(
            z,
            0.13f,
            (Color){ 70, 110, 230, 255 }
        );
    }

    if (gTool == TOOL_SCALE)
    {
        OBB box =
            GetPartOBB(p);

        Vector3 x =
            Vector3Add(
                p->position,
                Vector3Scale(
                    box.axis[0],
                    box.half.x + 0.55f
                )
            );

        Vector3 y =
            Vector3Add(
                p->position,
                Vector3Scale(
                    box.axis[1],
                    box.half.y + 0.55f
                )
            );

        Vector3 z =
            Vector3Add(
                p->position,
                Vector3Scale(
                    box.axis[2],
                    box.half.z + 0.55f
                )
            );

        DrawLine3D(
            p->position,
            x,
            (Color){ 225, 80, 80, 255 }
        );

        DrawLine3D(
            p->position,
            y,
            (Color){ 80, 205, 100, 255 }
        );

        DrawLine3D(
            p->position,
            z,
            (Color){ 80, 120, 235, 255 }
        );

        DrawCube(
            x,
            0.18f,
            0.18f,
            0.18f,
            (Color){ 225, 80, 80, 255 }
        );

        DrawCube(
            y,
            0.18f,
            0.18f,
            0.18f,
            (Color){ 80, 205, 100, 255 }
        );

        DrawCube(
            z,
            0.18f,
            0.18f,
            0.18f,
            (Color){ 80, 120, 235, 255 }
        );
    }
}

/* ------------------------------------------------------------ */
/* World rendering                                                */
/* ------------------------------------------------------------ */

static void DrawWorldDecoration(void)
{
    DrawPlane(
        V3(0, -0.015f, 0),
        V2(100, 100),
        (Color){
            184,
            188,
            193,
            255
        }
    );

    for (int i = -50;
         i <= 50;
         i += 10)
    {
        DrawLine3D(
            V3(
                (float)i,
                0.006f,
                -50
            ),
            V3(
                (float)i,
                0.006f,
                50
            ),
            (Color){
                164,
                168,
                173,
                60
            }
        );

        DrawLine3D(
            V3(
                -50,
                0.006f,
                (float)i
            ),
            V3(
                50,
                0.006f,
                (float)i
            ),
            (Color){
                164,
                168,
                173,
                60
            }
        );
    }
}

static void DrawPartIndicators(
    const Part *p
)
{
    if (!p->used)
        return;

    if (p->motor != MOTOR_NONE)
    {
        DrawSphere(
            Vector3Add(
                p->position,
                V3(
                    0,
                    PartRadius(p) +
                    0.25f,
                    0
                )
            ),
            0.13f,
            (Color){
                245,
                190,
                40,
                255
            }
        );
    }

    if (gRunning &&
        p->sleeping &&
        p->type != PART_BASE)
    {
        DrawSphere(
            Vector3Add(
                p->position,
                V3(
                    0,
                    PartRadius(p) +
                    0.12f,
                    0
                )
            ),
            0.08f,
            (Color){
                125,
                145,
                160,
                230
            }
        );
    }
}

static void Draw3DWorld(
    EditorCamera *editor
)
{
    DrawWorldDecoration();

    if (gPartShader.id > 0)
    {
        Vector3 light =
            V3(
                -0.45f,
                -0.9f,
                -0.35f
            );

        Vector3 cameraPosition =
            editor->camera.position;

        Vector4 fogColor =
            (Vector4){
                125.0f / 255.0f,
                170.0f / 255.0f,
                215.0f / 255.0f,
                1
            };

        if (gShaderLightLoc >= 0)
        {
            SetShaderValue(
                gPartShader,
                gShaderLightLoc,
                &light,
                SHADER_UNIFORM_VEC3
            );
        }

        if (gShaderCameraLoc >= 0)
        {
            SetShaderValue(
                gPartShader,
                gShaderCameraLoc,
                &cameraPosition,
                SHADER_UNIFORM_VEC3
            );
        }

        if (gShaderFogLoc >= 0)
        {
            SetShaderValue(
                gPartShader,
                gShaderFogLoc,
                &fogColor,
                SHADER_UNIFORM_VEC4
            );
        }

        BeginShaderMode(
            gPartShader
        );

        for (int i = 0;
             i < MAX_PARTS;
             i++)
        {
            if (!gParts[i].used)
                continue;

            DrawPartGeometry(
                &gParts[i]
            );
        }

        EndShaderMode();
    }
    else
    {
        for (int i = 0;
             i < MAX_PARTS;
             i++)
        {
            if (!gParts[i].used)
                continue;

            DrawPartGeometry(
                &gParts[i]
            );
        }
    }

    for (int i = 0;
         i < MAX_PARTS;
         i++)
    {
        if (!gParts[i].used)
            continue;

        DrawPartIndicators(
            &gParts[i]
        );
    }

    if (gSelected >= 0 &&
        gParts[gSelected].used)
    {
        DrawSelection(
            &gParts[gSelected]
        );
    }
}

/* ------------------------------------------------------------ */
/* Fonts                                                          */
/* ------------------------------------------------------------ */

static void LoadUIFonts(void)
{
    gUIFont =
        LoadFontEx(
            "C:/Windows/Fonts/tahoma.ttf",
            20,
            NULL,
            0
        );

    gUIBoldFont =
        LoadFontEx(
            "C:/Windows/Fonts/tahomabd.ttf",
            20,
            NULL,
            0
        );

    gUIFontCustom =
        gUIFont.texture.id != 0;

    gUIBoldFontCustom =
        gUIBoldFont.texture.id != 0;

    if (!gUIFontCustom)
        gUIFont =
            GetFontDefault();

    if (!gUIBoldFontCustom)
        gUIBoldFont =
            GetFontDefault();

    if (gUIFontCustom)
    {
        SetTextureFilter(
            gUIFont.texture,
            TEXTURE_FILTER_BILINEAR
        );
    }

    if (gUIBoldFontCustom)
    {
        SetTextureFilter(
            gUIBoldFont.texture,
            TEXTURE_FILTER_BILINEAR
        );
    }
}

static void UI_Text(
    const char *text,
    float x,
    float y,
    float size,
    Color color
)
{
    DrawTextEx(
        gUIFont,
        text,
        V2(x, y),
        size,
        0,
        color
    );
}

static void UI_Bold(
    const char *text,
    float x,
    float y,
    float size,
    Color color
)
{
    DrawTextEx(
        gUIBoldFont,
        text,
        V2(x, y),
        size,
        0,
        color
    );
}

/* ------------------------------------------------------------ */
/* UI button                                                      */
/* ------------------------------------------------------------ */

static void DrawButton(
    Rectangle rect,
    const char *text,
    bool active
)
{
    Color face =
        active
        ? (Color){
            190,
            207,
            229,
            255
        }
        : (Color){
            205,
            205,
            205,
            255
        };

    DrawRectangleRec(
        rect,
        face
    );

    DrawLine(
        rect.x,
        rect.y,
        rect.x +
        rect.width -
        1,
        rect.y,
        WHITE
    );

    DrawLine(
        rect.x,
        rect.y,
        rect.x,
        rect.y +
        rect.height -
        1,
        WHITE
    );

    DrawLine(
        rect.x,
        rect.y +
        rect.height -
        1,
        rect.x +
        rect.width -
        1,
        rect.y +
        rect.height -
        1,
        (Color){
            90,
            90,
            90,
            255
        }
    );

    DrawLine(
        rect.x +
        rect.width -
        1,
        rect.y,
        rect.x +
        rect.width -
        1,
        rect.y +
        rect.height -
        1,
        (Color){
            90,
            90,
            90,
            255
        }
    );

    Vector2 textSize =
        MeasureTextEx(
            gUIFont,
            text,
            14,
            0
        );

    UI_Text(
        text,
        rect.x +
        (rect.width -
         textSize.x) *
        0.5f,
        rect.y +
        (rect.height -
         textSize.y) *
        0.5f,
        14,
        BLACK
    );
}

/* ------------------------------------------------------------ */
/* Title bar                                                      */
/* ------------------------------------------------------------ */

static void DrawTitleBar(void)
{
    DrawRectangle(
        0,
        0,
        GetScreenWidth(),
        36,
        (Color){
            36,
            82,
            145,
            255
        }
    );

    DrawRectangle(
        0,
        36,
        GetScreenWidth(),
        2,
        (Color){
            15,
            40,
            70,
            255
        }
    );

    UI_Bold(
        "SimPhycs Alpha",
        10,
        8,
        17,
        WHITE
    );

    char state[96];

    snprintf(
        state,
        sizeof(state),
        "%s   |   %s [%s]",
        gRunning
            ? "RUNNING"
            : "EDIT MODE",
        ToolName(gTool),
        AxisName(gAxis)
    );

    Vector2 size =
        MeasureTextEx(
            gUIFont,
            state,
            13,
            0
        );

    UI_Text(
        state,
        GetScreenWidth() -
        size.x -
        14,
        10,
        13,
        WHITE
    );
}

/* ------------------------------------------------------------ */
/* Toolbar                                                        */
/* ------------------------------------------------------------ */

static void DrawToolbar(void)
{
    DrawRectangle(
        0,
        38,
        GetScreenWidth(),
        58,
        (Color){
            205,
            205,
            205,
            255
        }
    );

    const char *tools[] =
    {
        "Select",
        "Move",
        "Rotate",
        "Scale",
        "Force"
    };

    for (int i = 0;
         i < 5;
         i++)
    {
        DrawButton(
            (Rectangle){
                10 +
                i * 82,
                48,
                76,
                38
            },
            tools[i],
            gTool == i
        );
    }

    float x = 440;

    DrawButton(
        (Rectangle){
            x,
            48,
            78,
            38
        },
        "Block",
        false
    );

    DrawButton(
        (Rectangle){
            x + 82,
            48,
            78,
            38
        },
        "Ball",
        false
    );

    DrawButton(
        (Rectangle){
            x + 164,
            48,
            88,
            38
        },
        "Cylinder",
        false
    );

    DrawButton(
        (Rectangle){
            x + 256,
            48,
            78,
            38
        },
        "Base",
        false
    );

    int right =
        GetScreenWidth() -
        10;

    DrawButton(
        (Rectangle){
            right - 330,
            48,
            76,
            38
        },
        "Run",
        gRunning
    );

    DrawButton(
        (Rectangle){
            right - 244,
            48,
            76,
            38
        },
        "Stop",
        !gRunning
    );

    DrawButton(
        (Rectangle){
            right - 158,
            48,
            76,
            38
        },
        "Reset",
        false
    );

    DrawButton(
        (Rectangle){
            right - 72,
            48,
            72,
            38
        },
        "Save",
        false
    );

    if (!IsMouseButtonPressed(
            MOUSE_BUTTON_LEFT))
        return;

    Vector2 mouse =
        GetMousePosition();

    for (int i = 0;
         i < 5;
         i++)
    {
        Rectangle rect =
        {
            10 +
            i * 82,
            48,
            76,
            38
        };

        if (CheckCollisionPointRec(
                mouse,
                rect))
        {
            gTool =
                (ToolType)i;

            gAxis =
                AXIS_FREE;

            gMouseDragging = false;

            return;
        }
    }

    Rectangle block =
    {
        x,
        48,
        78,
        38
    };

    Rectangle ball =
    {
        x + 82,
        48,
        78,
        38
    };

    Rectangle cylinder =
    {
        x + 164,
        48,
        88,
        38
    };

    Rectangle base =
    {
        x + 256,
        48,
        78,
        38
    };

    if (CheckCollisionPointRec(
            mouse,
            block))
    {
        NewBlock();
        return;
    }

    if (CheckCollisionPointRec(
            mouse,
            ball))
    {
        NewBall();
        return;
    }

    if (CheckCollisionPointRec(
            mouse,
            cylinder))
    {
        NewCylinder();
        return;
    }

    if (CheckCollisionPointRec(
            mouse,
            base))
    {
        NewBase();
        return;
    }

    Rectangle run =
    {
        right - 330,
        48,
        76,
        38
    };

    Rectangle stop =
    {
        right - 244,
        48,
        76,
        38
    };

    Rectangle reset =
    {
        right - 158,
        48,
        76,
        38
    };

    if (CheckCollisionPointRec(
            mouse,
            run))
    {
        StartSimulation();
        return;
    }

    if (CheckCollisionPointRec(
            mouse,
            stop))
    {
        StopSimulation();
        return;
    }

    if (CheckCollisionPointRec(
            mouse,
            reset))
    {
        ResetEditorScene();
        return;
    }
}

/* ------------------------------------------------------------ */
/* Explorer                                                       */
/* ------------------------------------------------------------ */

static void DrawExplorer(void)
{
    if (!gShowExplorer)
        return;

    float x = 10;
    float y = 108;

    float width = 260;

    float height =
        GetScreenHeight() -
        160;

    DrawRectangle(
        x,
        y,
        width,
        height,
        (Color){
            225,
            225,
            225,
            255
        }
    );

    DrawRectangle(
        x,
        y,
        width,
        30,
        (Color){
            175,
            175,
            180,
            255
        }
    );

    UI_Bold(
        "Explorer",
        x + 9,
        y + 6,
        15,
        BLACK
    );

    float row =
        y + 42;

    UI_Text(
        "Workspace",
        x + 12,
        row,
        14,
        (Color){
            50,
            50,
            50,
            255
        }
    );

    row += 27;

    for (int i = 0;
         i < MAX_PARTS;
         i++)
    {
        if (!gParts[i].used)
            continue;

        if (i == gSelected)
        {
            DrawRectangle(
                x + 5,
                row - 3,
                width - 10,
                25,
                (Color){
                    185,
                    205,
                    230,
                    255
                }
            );
        }

        char text[100];

        snprintf(
            text,
            sizeof(text),
            "%s  [%s]",
            gParts[i].name,
            PartTypeName(
                gParts[i].type
            )
        );

        UI_Text(
            text,
            x + 24,
            row,
            13,
            i == gSelected
            ? (Color){
                20,
                70,
                145,
                255
            }
            : BLACK
        );

        row += 26;

        if (row >
            y + height - 25)
            break;
    }
}

/* ------------------------------------------------------------ */
/* Properties                                                     */
/* ------------------------------------------------------------ */

static void DrawProperties(void)
{
    if (!gShowProperties)
        return;

    float width = 305;

    float x =
        GetScreenWidth() -
        width -
        10;

    float y = 108;

    float height =
        GetScreenHeight() -
        160;

    DrawRectangle(
        x,
        y,
        width,
        height,
        (Color){
            225,
            225,
            225,
            255
        }
    );

    DrawRectangle(
        x,
        y,
        width,
        30,
        (Color){
            175,
            175,
            180,
            255
        }
    );

    UI_Bold(
        "Properties",
        x + 9,
        y + 6,
        15,
        BLACK
    );

    if (gSelected < 0 ||
        !gParts[gSelected].used)
    {
        UI_Text(
            "No selection",
            x + 12,
            y + 48,
            14,
            (Color){
                90,
                90,
                90,
                255
            }
        );

        return;
    }

    Part *p =
        &gParts[gSelected];

    float py =
        y + 43;

    char text[180];

    snprintf(
        text,
        sizeof(text),
        "Name: %s",
        p->name
    );

    UI_Bold(
        text,
        x + 10,
        py,
        14,
        BLACK
    );

    py += 27;

    snprintf(
        text,
        sizeof(text),
        "Type: %s",
        PartTypeName(
            p->type
        )
    );

    UI_Text(
        text,
        x + 10,
        py,
        13,
        BLACK
    );

    py += 28;

    UI_Bold(
        "Transform",
        x + 10,
        py,
        14,
        (Color){
            30,
            70,
            125,
            255
        }
    );

    py += 23;

    snprintf(
        text,
        sizeof(text),
        "Position  %.2f  %.2f  %.2f",
        p->position.x,
        p->position.y,
        p->position.z
    );

    UI_Text(
        text,
        x + 10,
        py,
        12,
        BLACK
    );

    py += 20;

    snprintf(
        text,
        sizeof(text),
        "Rotation  %.1f  %.1f  %.1f",
        p->rotation.x *
            RAD2DEG_F,
        p->rotation.y *
            RAD2DEG_F,
        p->rotation.z *
            RAD2DEG_F
    );

    UI_Text(
        text,
        x + 10,
        py,
        12,
        BLACK
    );

    py += 20;

    snprintf(
        text,
        sizeof(text),
        "Size      %.2f  %.2f  %.2f",
        p->size.x,
        p->size.y,
        p->size.z
    );

    UI_Text(
        text,
        x + 10,
        py,
        12,
        BLACK
    );

    py += 30;

    UI_Bold(
        "Physics",
        x + 10,
        py,
        14,
        (Color){
            30,
            70,
            125,
            255
        }
    );

    py += 23;

    snprintf(
        text,
        sizeof(text),
        "Mass       %.2f",
        p->mass
    );

    UI_Text(
        text,
        x + 10,
        py,
        12,
        BLACK
    );

    py += 20;

    snprintf(
        text,
        sizeof(text),
        "Friction   %.2f",
        p->friction
    );

    UI_Text(
        text,
        x + 10,
        py,
        12,
        BLACK
    );

    py += 20;

    snprintf(
        text,
        sizeof(text),
        "Bounce     %.2f",
        p->bounce
    );

    UI_Text(
        text,
        x + 10,
        py,
        12,
        BLACK
    );

    py += 25;

    snprintf(
        text,
        sizeof(text),
        "Anchored   %s",
        p->anchored
        ? "Yes"
        : "No"
    );

    UI_Text(
        text,
        x + 10,
        py,
        12,
        BLACK
    );

    py += 19;

    snprintf(
        text,
        sizeof(text),
        "Frozen     %s",
        p->frozen
        ? "Yes"
        : "No"
    );

    UI_Text(
        text,
        x + 10,
        py,
        12,
        BLACK
    );

    py += 19;

    snprintf(
        text,
        sizeof(text),
        "Collisions %s",
        p->collisions
        ? "On"
        : "Off"
    );

    UI_Text(
        text,
        x + 10,
        py,
        12,
        BLACK
    );

    py += 29;

    UI_Bold(
        "Motor",
        x + 10,
        py,
        14,
        (Color){
            30,
            70,
            125,
            255
        }
    );

    py += 23;

    snprintf(
        text,
        sizeof(text),
        "Type       %s",
        MotorName(
            p->motor
        )
    );

    UI_Text(
        text,
        x + 10,
        py,
        12,
        BLACK
    );

    py += 20;

    snprintf(
        text,
        sizeof(text),
        "Power      %.2f",
        p->motorPower
    );

    UI_Text(
        text,
        x + 10,
        py,
        12,
        BLACK
    );

    py += 30;

    UI_Bold(
        "Editor",
        x + 10,
        py,
        14,
        (Color){
            30,
            70,
            125,
            255
        }
    );

    py += 23;

    UI_Text(
        "X / Y / Z   Axis",
        x + 10,
        py,
        12,
        BLACK
    );

    py += 18;

    UI_Text(
        "M           Spin",
        x + 10,
        py,
        12,
        BLACK
    );

    py += 18;

    UI_Text(
        "P           Push",
        x + 10,
        py,
        12,
        BLACK
    );

    py += 18;

    UI_Text(
        "B           Conveyor",
        x + 10,
        py,
        12,
        BLACK
    );

    py += 18;

    UI_Text(
        "C           Duplicate",
        x + 10,
        py,
        12,
        BLACK
    );

    py += 18;

    UI_Text(
        "Delete      Remove",
        x + 10,
        py,
        12,
        BLACK
    );

    py += 26;

    UI_Bold(
        "Movement",
        x + 10,
        py,
        14,
        (Color){
            30,
            70,
            125,
            255
        }
    );

    py += 22;

    UI_Text(
        "WASD  Camera",
        x + 10,
        py,
        11,
        BLACK
    );

    py += 17;

    UI_Text(
        "Move + wheel = up/down",
        x + 10,
        py,
        11,
        BLACK
    );

    py += 17;

    UI_Text(
        "PageUp/PageDn = Y",
        x + 10,
        py,
        11,
        BLACK
    );

    py += 17;

    UI_Text(
        "Scale + Ctrl = uniform",
        x + 10,
        py,
        11,
        BLACK
    );
}

/* ------------------------------------------------------------ */
/* Bottom bar                                                     */
/* ------------------------------------------------------------ */

static void DrawBottomBar(void)
{
    int height = 44;

    int y =
        GetScreenHeight() -
        height;

    DrawRectangle(
        0,
        y,
        GetScreenWidth(),
        height,
        (Color){
            205,
            205,
            205,
            255
        }
    );

    DrawRectangleLines(
        0,
        y,
        GetScreenWidth(),
        height,
        (Color){
            90,
            90,
            90,
            255
        }
    );

    UI_Text(
        "Q Select   W Move   E Rotate   R Scale   F Force   "
        "X/Y/Z Axis   1 Block   2 Ball   3 Cylinder   Space Run",
        10,
        y + 14,
        12,
        BLACK
    );

    char fps[32];

    snprintf(
        fps,
        sizeof(fps),
        "%d FPS",
        GetFPS()
    );

    Vector2 size =
        MeasureTextEx(
            gUIFont,
            fps,
            12,
            0
        );

    UI_Text(
        fps,
        GetScreenWidth() -
        size.x -
        12,
        y + 14,
        12,
        BLACK
    );
}

/* ------------------------------------------------------------ */
/* Keyboard                                                       */
/* ------------------------------------------------------------ */

static void HandleKeyboard(void)
{
    /*
        W is intentionally NOT used as a tool shortcut anymore.
        WASD owns camera movement.

        Move is still available from the toolbar.
        This prevents the old W/camera conflict.
    */
    if (IsKeyPressed(KEY_Q))
    {
        gTool = TOOL_SELECT;
        gAxis = AXIS_FREE;
    }

    if (IsKeyPressed(KEY_E))
    {
        gTool = TOOL_ROTATE;
        gAxis = AXIS_FREE;
    }

    if (IsKeyPressed(KEY_R))
    {
        gTool = TOOL_SCALE;
        gAxis = AXIS_FREE;
    }

    if (IsKeyPressed(KEY_F))
    {
        gTool = TOOL_FORCE;
        gAxis = AXIS_FREE;
    }

    if (IsKeyPressed(KEY_X))
        gAxis = AXIS_X;

    if (IsKeyPressed(KEY_Y))
        gAxis = AXIS_Y;

    if (IsKeyPressed(KEY_Z))
        gAxis = AXIS_Z;

    if (IsKeyPressed(KEY_ONE))
        NewBlock();

    if (IsKeyPressed(KEY_TWO))
        NewBall();

    if (IsKeyPressed(KEY_THREE))
        NewCylinder();

    if (IsKeyPressed(KEY_FOUR))
        NewBase();

    if (IsKeyPressed(KEY_SPACE))
    {
        if (gRunning)
            StopSimulation();
        else
            StartSimulation();
    }

    if (IsKeyPressed(KEY_DELETE))
        DeleteSelected();

    if (IsKeyPressed(KEY_C))
        DuplicateSelected();

    if (IsKeyPressed(KEY_F5))
        ResetEditorScene();

    if (IsKeyPressed(KEY_F1))
        gShowExplorer =
            !gShowExplorer;

    if (IsKeyPressed(KEY_F2))
        gShowProperties =
            !gShowProperties;

    if (IsKeyPressed(KEY_ESCAPE))
    {
        gSelected = -1;
        gMouseDragging = false;
        gAxis = AXIS_FREE;
    }

    /*
        Move tool is selected from toolbar or by clicking
        the visual Move button. This avoids breaking WASD.
    */

    if (gSelected >= 0 &&
        gParts[gSelected].used)
    {
        Part *p =
            &gParts[gSelected];

        if (IsKeyPressed(KEY_M))
        {
            p->motor =
                p->motor == MOTOR_SPIN
                ? MOTOR_NONE
                : MOTOR_SPIN;
        }

        if (IsKeyPressed(KEY_P))
        {
            p->motor =
                p->motor == MOTOR_PUSH
                ? MOTOR_NONE
                : MOTOR_PUSH;
        }

        if (IsKeyPressed(KEY_B))
        {
            p->motor =
                p->motor == MOTOR_CONVEYOR
                ? MOTOR_NONE
                : MOTOR_CONVEYOR;
        }
    }
}

/* ------------------------------------------------------------ */
/* Main                                                           */
/* ------------------------------------------------------------ */

int main(void)
{
    SetConfigFlags(
        FLAG_WINDOW_RESIZABLE |
        FLAG_MSAA_4X_HINT |
        FLAG_VSYNC_HINT
    );

    InitWindow(
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        "SimPhycs Alpha"
    );

    SetTargetFPS(120);

    LoadUIFonts();

    CreatePartShader();

    EditorCamera editor;

    memset(
        &editor,
        0,
        sizeof(editor)
    );

    editor.target =
        V3(0, 2, 0);

    editor.yaw =
        0.65f;

    editor.pitch =
        0.55f;

    editor.distance =
        32.0f;

    editor.camera.fovy =
        55.0f;

    editor.camera.projection =
        CAMERA_PERSPECTIVE;

    editor.camera.up =
        V3(0, 1, 0);

    UpdateEditorCamera(
        &editor
    );

    CreateInitialScene();

    while (!WindowShouldClose())
    {
        float frameTime =
            GetFrameTime();

        frameTime =
            ClampF(
                frameTime,
                0.0f,
                0.1f
            );

        HandleKeyboard();

        UpdateCameraInput(
            &editor
        );

        if (!gRunning)
        {
            UpdateEditorMouse(
                &editor
            );

            UpdateKeyboardTools();
        }

        /*
            240 Hz fixed physics with a maximum number of
            substeps per rendered frame.

            This is substantially more stable than doing only
            one physics step per frame.
        */
        if (gRunning)
        {
            gPhysicsAccumulator +=
                frameTime;

            if (gPhysicsAccumulator >
                0.25f)
            {
                gPhysicsAccumulator =
                    0.25f;
            }

            int substeps = 0;

            while (
                gPhysicsAccumulator >=
                PHYSICS_DT &&
                substeps <
                MAX_SUBSTEPS_PER_FRAME
            )
            {
                PhysicsStep(
                    PHYSICS_DT
                );

                gPhysicsAccumulator -=
                    PHYSICS_DT;

                substeps++;
            }
        }

        BeginDrawing();

        ClearBackground(
            (Color){
                125,
                170,
                215,
                255
            }
        );

        BeginMode3D(
            editor.camera
        );

        Draw3DWorld(
            &editor
        );

        EndMode3D();

        DrawTitleBar();

        DrawToolbar();

        DrawExplorer();

        DrawProperties();

        DrawBottomBar();

        EndDrawing();
    }

    for (int i = 0;
         i < MAX_PARTS;
         i++)
    {
        if (gParts[i].used)
            DestroyPart(
                &gParts[i]
            );
    }

    if (gPartShader.id > 0)
    {
        UnloadShader(
            gPartShader
        );
    }

    if (gUIFontCustom)
        UnloadFont(gUIFont);

    if (gUIBoldFontCustom)
        UnloadFont(gUIBoldFont);

    CloseWindow();

    return 0;
}