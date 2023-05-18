using OpenDis.Dis2012;
using System;
using UnityEngine;

public class REFERENCE_FRAMES
{
	static double M_PI = 3.1415926535897932;	// rads
	static double M_PI_2 = 1.570796326794897;   // rads
    static double a = 6378137.0;				// Earth semi-major axis
	static double b = 6356752.314245;			// Earth semi-minor axis
    public static double DEG_TO_RAD = M_PI / 180;

    // ONE_SECOND is pi/180/60/60, or about 100 feet at earths' equator
    static double ONE_SECOND = 4.848136811E-6;
	//static double EQUATORIAL_RADIUS_FT = 20925650.0;		// ft
	public static double EQUATORIAL_RADIUS_M = 6378137.0;	// meters
	// E  = 1-f
	// EPS = sqrt(1-(1-f)^2)
	static double E = 0.996647189335253;
	static double EPS = 0.081819190842621;
	static double RESQ_M = 40680645877797.1344;      // meters

    // Other vars
    //static double RESQ_FT = 437882827922500.0;     // ft
    //static double DELTA = 0.000000009;
    //static double FG_PI = 3.141592653589793238462643383279502884197169399375105820975;
    //static double MPERDEG = 111120.0;


    ////////////////////////////////
    // GENERIC MATRICES FUNCTIONS //
    ////////////////////////////////

    /// This function evaluates the product beetween a 3x3 matrices and a 3x1 vector
    /// Params:
    /// double matr[3][3] - a 3x3 matrix as first factor
    /// double vect[3] - a 3x1 vector as second factor
    /// Return value:
    /// double prod[3] - a 3x1 vector as a result of matr * vect
    public static double[] prodMatrVect3(double[,] matr, double[] vect)
    {
        double[] prod = new double[3];

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
				prod[i] += matr[i, j] * vect[j];
        return prod;
    }

    /// This function evaluates the product beetween two 3x3 matrices
    /// Params:
    /// double matr1[3][3] - a 3x3 matrix as first factor
    /// double matr2[3][3] - a 3x3 matrix as second factor
    /// double prodMatrix[3][3] - a 3x3 matrix as a result of matr1 * matr2
    /// Return value:
    /// none
    public static double[,] prodMatr3(double[,] matr1, double[,] matr2)
	{
		double[,] prodMatrix = new double[3, 3];

		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				for (int k = 0; k < 3; k++)
					prodMatrix[i, j] += matr1[i, k] * matr2[k, j];
		return prodMatrix;
	}


	/// This function evaluate the transpose of a 3x3 matrix
	/// Params:
	/// double matr[3][3] a 3x3 matrix as input
	/// double transposeMatrix[3][3] a 3x3 transposed matrix as a result
	/// Return value:
	/// none
	public static double[,] transposeMatr3(double[,] matr)
	{
		double[,] transposeMatrix = new double[3, 3];

		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				transposeMatrix[j, i] = matr[i, j];
		return transposeMatrix;
	}


    /// This function evaluates a generic rotation matrix
    /// Params:
    /// int axis 1, 2, 3: numbers of the axes on which the rotations are made
    /// double angle1, 2, 3: amplitude of angles of respective axes on which the rotations are made [RAD]
    /// Return value:
    /// double[,] rotation matrix for the 3 rotations
    public static double[,] matRGeneral(int axis1, int axis2, int axis3, double angle1, double angle2, double angle3)
	{
		double[,] mat1, mat2, mat3, matR;

        // First rotation
        if (axis1 == 1)
			mat1 = new double[,] { { 1, 0, 0 }, { 0, Math.Cos(angle1), Math.Sin(angle1) }, { 0, -Math.Sin(angle1), Math.Cos(angle1) } };
		else if (axis1 == 2)
			mat1 = new double[,] { { Math.Cos(angle1), 0, -Math.Sin(angle1) }, { 0, 1, 0 }, { Math.Sin(angle1), 0, Math.Cos(angle1) } };
		else if (axis1 == 3)
			mat1 = new double[,] { { Math.Cos(angle1), Math.Sin(angle1), 0 }, { -Math.Sin(angle1), Math.Cos(angle1), 0 }, { 0, 0, 1 } };
        else
		{
            Debug.Log("axis1 not defined!");
            mat1 = new double[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
        }
        
        // Second rotation
        if (axis2 == 1)
            mat2 = new double[,] { { 1, 0, 0 }, { 0, Math.Cos(angle2), Math.Sin(angle2) }, { 0, -Math.Sin(angle2), Math.Cos(angle2) } };
        else if (axis2 == 2)
            mat2 = new double[,] { { Math.Cos(angle2), 0, -Math.Sin(angle2) }, { 0, 1, 0 }, { Math.Sin(angle2), 0, Math.Cos(angle2) } };
        else if (axis2 == 3)
            mat2 = new double[,] { { Math.Cos(angle2), Math.Sin(angle2), 0 }, { -Math.Sin(angle2), Math.Cos(angle2), 0 }, { 0, 0, 1 } };
        else
        {
            Debug.Log("axis2 not defined!");
            mat2 = new double[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
        }

        // Third rotation
        if (axis3 == 1)
			mat3 = new double[,] { { 1, 0, 0 }, { 0, Math.Cos(angle3), Math.Sin(angle3) }, { 0, -Math.Sin(angle3), Math.Cos(angle3) } };
		else if (axis3 == 2)
			mat3 = new double[,] { { Math.Cos(angle3), 0, -Math.Sin(angle3) }, { 0, 1, 0 }, { Math.Sin(angle3), 0, Math.Cos(angle3) } };
		else if (axis3 == 3)
			mat3 = new double[,] { { Math.Cos(angle3), Math.Sin(angle3), 0 }, { -Math.Sin(angle3), Math.Cos(angle3), 0 }, { 0, 0, 1 } };
        else
        {
            Debug.Log("axis3 not defined!");
            mat3 = new double[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
        }

        matR = prodMatr3(mat3, prodMatr3(mat2, mat1));

        return matR;
    }


    /////////////////////////////////////
    // COORDINATE CONVERSION FUNCTIONS //
    /////////////////////////////////////

    // Convert cartesian coordinates to polar coordinates. Lon and Lat
    // are specified in degrees.  Distances are specified in meters.
    public static (double lat, double lon, double alt) CartToGeod(double x, double y, double z)
	{
		double lat, lon, alt;
		double lat_geoc, radius, sea_level_r;
		//  lat_geoc=(PI_2 - atan2(sqrt(x*x+y*y),z))/DEG_TO_RAD;
		lat_geoc = Math.Atan2(z, Math.Sqrt((x*x+y*y))) / DEG_TO_RAD;
		lon = Math.Atan2(y, x) / DEG_TO_RAD;
		radius = Math.Sqrt((x *x + y*y + z*z));
		(lat, alt, sea_level_r) = GeocToGeod(lat_geoc, radius);

		return (lat, lon, alt);
	}


    // GeocToGeod(lat_geoc, radius, *lat_geod, *alt, *sea_level_r)
    //
    //     INPUT:
    //         lat_geoc	Geocentric latitude, in degrees, positive = North
    //         radius	C.G. radius to earth center (meters)
    //
    //     OUTPUT:
    //         lat_geod	Geodetic latitude, in degrees, positive = North
    //         alt		C.G. altitude above mean sea level (meters)
    //         sea_level_r	radius from earth center to sea level at
    //                      local vertical (surface normal) of C.G. (meters)
    //
    public static(double lat_geod, double alt, double sea_level_r) GeocToGeod( double lat_geoc, double radius)
	{
		double lat_geod, alt, sea_level_r;

        double t_lat, x_alpha, mu_alpha, delt_mu, r_alpha, l_point, rho_alpha;
	    double sin_mu_a, denom,delt_lambda, lambda_sl, sin_lambda_sl, tmp;

	    lat_geoc *= M_PI_2 / 90.0;
        // near North pole or near South pole
        if ( ( (M_PI_2 - lat_geoc) < ONE_SECOND ) || ( (M_PI_2 + lat_geoc) < ONE_SECOND ) )
	    {
			lat_geod = lat_geoc;
			sea_level_r = EQUATORIAL_RADIUS_M*E;
			alt = radius - sea_level_r;
	    } 
		else 
		{
			// cout << "  lat_geoc = " << lat_geoc << endl;
			t_lat = Math.Tan(lat_geoc);
			// cout << "  tan(t_lat) = " << t_lat << endl;
			x_alpha = E*EQUATORIAL_RADIUS_M/ Math.Sqrt((t_lat *t_lat + E*E));
			// cout << "  x_alpha = " << x_alpha << endl;
			tmp = RESQ_M - x_alpha * x_alpha;
			if ( tmp < 0.0 ) { tmp = 0.0; }
			mu_alpha = Math.Atan2(Math.Sqrt(tmp),(E*x_alpha));
			if (lat_geoc < 0) mu_alpha = - mu_alpha;
			sin_mu_a = Math.Sin(mu_alpha);
			delt_lambda = mu_alpha - lat_geoc;
			r_alpha = x_alpha/ Math.Cos(lat_geoc);
			l_point = radius - r_alpha;
			alt = l_point*Math.Cos(delt_lambda);

			denom = Math.Sqrt((1-EPS*EPS*sin_mu_a*sin_mu_a));
			rho_alpha = EQUATORIAL_RADIUS_M*(1-EPS)/
				(denom*denom*denom);
			delt_mu = Math.Atan2((l_point* Math.Sin(delt_lambda)),(rho_alpha + alt));
			lat_geod = mu_alpha - delt_mu;
			lambda_sl = Math.Atan((E*E * Math.Tan(lat_geod))); // SL geoc. latitude
			sin_lambda_sl = Math.Sin(lambda_sl );
			sea_level_r = Math.Sqrt((RESQ_M / (1 + ((1/(E*E))-1)*sin_lambda_sl*sin_lambda_sl)));
	    }
	    lat_geod *= 90.0 / M_PI_2;

		return (lat_geod, alt, sea_level_r);
    }


    // GeodToGeoc( lat_geod, alt, *sl_radius, *lat_geoc )
    //
    //     INPUT:
    //         lat_geod	Geodetic latitude, in degrees, positive = North
    //         alt		C.G. altitude above mean sea level (in meters)
    //
    //     OUTPUT:
    //         sl_radius	SEA LEVEL radius to earth center (in meters)
    //                      (add Altitude to get true distance from earth center.
    //         lat_geoc	Geocentric latitude, in degrees, positive = North
    //
    public static(double sl_radius, double lat_geoc) GeodToGeoc( double lat_geod, double alt)
	{
        double sl_radius, lat_geoc;
        double lambda_sl, sin_lambda_sl, cos_lambda_sl, sin_mu, cos_mu, px, py;

	    lat_geod *= M_PI_2 / 90.0;
	    lambda_sl = Math.Atan(( E*E * Math.Tan(lat_geod) )); // sea level geocentric latitude
	    sin_lambda_sl = Math.Sin(lambda_sl );
	    cos_lambda_sl = Math.Cos(lambda_sl );
	    sin_mu = Math.Sin(lat_geod);                  // Geodetic (map makers') latitude
	    cos_mu = Math.Cos(lat_geod);
	    sl_radius = Math.Sqrt((RESQ_M / (1 + ((1/(E*E))-1) * sin_lambda_sl * sin_lambda_sl)));
	    py = sl_radius * sin_lambda_sl + alt * sin_mu;
	    px = sl_radius * cos_lambda_sl + alt * cos_mu;
	    lat_geoc = Math.Atan2(py, px ) * 90.0 / M_PI_2;

		return (sl_radius, lat_geoc);
	}

    // This function evaluates ECEF xyz position from WGS84 lat, lon, alt
    public static double[] wgs84_to_ecef(double lat, double lon, double alt)
    {
        double[] ecef_position = new double[3];

        //Conversion from WGS84 to ECEF (Earth Centered Earth Fixed)
        ecef_position[0] = (a / (Math.Sqrt(Math.Pow(Math.Cos(lat), 2) + Math.Pow(b, 2) / Math.Pow(a, 2) * Math.Pow(Math.Sin(lat), 2))) + alt) * Math.Cos(lat) * Math.Cos(lon);
        ecef_position[1] = (a / (Math.Sqrt(Math.Pow(Math.Cos(lat), 2) + Math.Pow(b, 2) / Math.Pow(a, 2) * Math.Pow(Math.Sin(lat), 2))) + alt) * Math.Cos(lat) * Math.Sin(lon);
        ecef_position[2] = (b / (Math.Sqrt(Math.Pow(Math.Sin(lat), 2) + Math.Pow(a, 2) / Math.Pow(b, 2) * Math.Pow(Math.Cos(lat), 2))) + alt) * Math.Sin(lat);

        return ecef_position;
    }


    //////////////////////////////////////
    // VECTORS TRANSFORMATION FUNCTIONS //
    //////////////////////////////////////

    // 123 vector transform function, TO TEST
    public static double[] vector_transform_123(float vector_x_in, float vector_y_in, float vector_z_in, float angle3, float angle2, float angle1)
    {
        double[] vector_out = new double[3];

        //Apply rotation around Z axis
        double dxp = vector_x_in * Math.Cos(-angle3) + vector_y_in * Math.Sin(-angle3);
        double dyp = -vector_x_in * Math.Sin(-angle3) + vector_y_in * Math.Cos(-angle3);
        double dzp = -vector_z_in;

        //Apply rotation around Y axis
        double dxs = dxp * Math.Cos(-angle2) - dzp * Math.Sin(-angle2);
        double dys = dyp;
        double dzs = dxp * Math.Sin(-angle2) + dzp * Math.Cos(-angle2);

        //Apply rotation around X axis
        vector_out[0] = dxs;
        vector_out[1] = dys * Math.Cos(-angle1) + dzs * Math.Sin(-angle1);
        vector_out[2] = -dys * Math.Sin(-angle1) + dzs * Math.Cos(-angle1);

        return vector_out;
    }

    // 312 vector transform function, TO TEST
    public static double[] vector_transform_312(float vector_x_in, float vector_y_in, float vector_z_in, float angle2, float angle1, float angle3)
    {
        double[] vector_out = new double[3];

        //Apply rotation around Y axis
        double dxp = vector_x_in * Math.Cos(-angle2) - vector_z_in * Math.Sin(-angle2);
        double dyp = vector_y_in;
        double dzp = vector_x_in * Math.Sin(-angle2) + vector_z_in * Math.Cos(-angle2);

        //Apply rotation around X axis
        double dxs = dxp;
        double dys = dyp * Math.Cos(-angle1) + dzp * Math.Sin(-angle1);
        double dzs = -dyp * Math.Sin(-angle1) + dzp * Math.Cos(-angle1);

		//Apply rotation around Z axis
		vector_out[0] = dxs * Math.Cos(-angle3) + dys * Math.Sin(-angle3);
        vector_out[1] = -dxs * Math.Sin(-angle3) + dys * Math.Cos(-angle3);
        vector_out[2] = -dzs;        

        return vector_out;
    }

    // Vector transform from Unity body Reference frame to ECEF
    public static double[] vector_transform_BodyU_ECEF(double lat_deg, double lon_deg, double[] eulerAngles, double[] vector_BodyU)
	{
		double[] vector_ECEF;
        //double[,] matr = transposeMatr3(ECEF_BodyU_MatR(lat_deg, 45, eulerAngles));
        double[,] matr = transposeMatr3(ECEF_BodyU_MatR(lat_deg, lon_deg, eulerAngles));
        vector_ECEF = prodMatrVect3(matr, vector_BodyU);
        //vector_ECEF[2] = -vector_ECEF[2];
        return vector_ECEF;
    }

    //Generic vector transform function, TO TEST
    static double[] vector_transform(double[] vector_in, int axis1, int axis2, int axis3, double angle1, double angle2, double angle3)
    {
        double[] vector_out;
        double[,] matr = matRGeneral(axis1, axis2, axis3, angle1, angle2, angle3);
        vector_out = prodMatrVect3(matr, vector_in);
        return vector_out;
    }


    //////////////////////////////////////////
    // ORIENTATION TRANSFORMATION FUNCTIONS //
    //////////////////////////////////////////

    // MATRIX CALCULATION FROM ONE REF TO ANOTHER FUNCTIONS

    // function that calculates the rotation matrix between ECEF reference frame and NED reference frame (32(1) lat and lon rotations)
    public static double[,] ECEF_NED_MatR(double lat_deg, double lon_deg)
    {
        double[,] R_ECEF_NED = matRGeneral(3, 2, 1, -lon_deg * DEG_TO_RAD, lat_deg * DEG_TO_RAD + M_PI / 2, 0);

        return R_ECEF_NED;
    }
    // function that calculates the rotation matrix between ECEF reference frame and ENU reference frame (32(1) lat and lon rotations)
    public static double[,] ECEF_ENU_MatR(double lat_deg, double lon_deg)
    {
        double[,] R_ECEF_ENU = matRGeneral(3, 2, 1, M_PI / 2 + lon_deg * DEG_TO_RAD, 0, M_PI / 2 - lat_deg * DEG_TO_RAD);

        return R_ECEF_ENU;
    }

    // function that calculates the rotation matrix between ECEF reference frame and BODY reference frame (321 tait-bryan angles (psi, theta, phi))
    public static double[,] BODY_ECEF_MatR(double[] TB_Angles_Deg)
    {
        //double[,] R_ECEF_BODY = transposeMatr3(matRGeneral(3, 2, 1, -TB_Angles_Deg[2] * DEG_TO_RAD, -TB_Angles_Deg[1] * DEG_TO_RAD, -TB_Angles_Deg[0] * DEG_TO_RAD));
        double[,] R_ECEF_BODY = matRGeneral(1, 2, 3, TB_Angles_Deg[0] * DEG_TO_RAD, TB_Angles_Deg[1] * DEG_TO_RAD, TB_Angles_Deg[2] * DEG_TO_RAD);

        return R_ECEF_BODY;
    }

    // function that calculates the rotation matrix between ECEF reference frame and BODY reference frame (321 tait-bryan angles (psi, theta, phi))
    public static double[,] ECEF_BODY_MatR(double[] TB_Angles_Deg)
    {
        //double[,] R_ECEF_BODY = matRGeneral(3, 2, 1, -TB_Angles_Deg[2] * DEG_TO_RAD, -TB_Angles_Deg[1] * DEG_TO_RAD, -TB_Angles_Deg[0] * DEG_TO_RAD);
        double[,] R_ECEF_BODY = transposeMatr3(matRGeneral(1, 2, 3, TB_Angles_Deg[0] * DEG_TO_RAD, TB_Angles_Deg[1] * DEG_TO_RAD, TB_Angles_Deg[2] * DEG_TO_RAD));

        return R_ECEF_BODY;
    }

    // function that calculates the rotation matrix between NED reference frame and BODY reference frame (321 tait-bryan angles (yaw, pitch, roll))
    public static double[,] NED_BODY_MatR(double[] TB_Angles_Deg)
    {
        double[,] R_NED_BODY = matRGeneral(3, 2, 1, -TB_Angles_Deg[0] * DEG_TO_RAD, -TB_Angles_Deg[1] * DEG_TO_RAD, -TB_Angles_Deg[2] * DEG_TO_RAD);

        return R_NED_BODY;
    }

    // UNITY FUNCTIONS

    // function that calculates the rotation matrix between Unity World reference frame and Unity Body reference frame (213 rotations)
    public static double[,] WorldU_BodyU_MatR(double[] eulerAngles_Deg)
    {
        double[,] R_WorldU_BodyU = matRGeneral(2, 1, 3, eulerAngles_Deg[1] * DEG_TO_RAD, eulerAngles_Deg[0] * DEG_TO_RAD, eulerAngles_Deg[2] * DEG_TO_RAD);

        return R_WorldU_BodyU;
    }

    // Function that calculates the rotation matrix between ECEF reference frame and Unity Body reference frame, assuming that the Unity World reference frame is a topocentric one with its lat and lon
    public static double[,] ECEF_BodyU_MatR(double lat_deg, double lon_deg, double[] eulerAngles_Deg)
    {
        // R_RANDOM = Matrix from the Unity body to the real body axes (front, right wing, down)
        double[,] R_RANDOM = new double[,] { { 0, 1, 0 }, { 0, 0, -1 }, { 1, 0, 0 } };

        double[,] R_WorldU_BodyU = matRGeneral(2, 1, 3, eulerAngles_Deg[1] * DEG_TO_RAD, eulerAngles_Deg[0] * DEG_TO_RAD, eulerAngles_Deg[2] * DEG_TO_RAD);
        double[,] R_BodyU_WorldU = prodMatr3(transposeMatr3(R_WorldU_BodyU), R_RANDOM);

        // Rotation matrix between Unity world and My NED
        double[,] R_WorldU_ENU = new double[,] { { 1, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 } };
        double[,] R_ENU_WorldU = transposeMatr3(R_WorldU_ENU);

        // Rotation matrix between My ENU and My ECEF
        double[,] R_ECEF_ENU = matRGeneral(3, 2, 1, M_PI / 2 + lon_deg * DEG_TO_RAD, 0, M_PI / 2 - lat_deg * DEG_TO_RAD);
        double[,] R_ENU_ECEF = transposeMatr3(R_ECEF_ENU);

        // Total rotation matrices between Unity body and My ECEF
        double[,] R_BodyU_ECEF = prodMatr3(R_ENU_ECEF, prodMatr3(R_WorldU_ENU, R_BodyU_WorldU));
        double[,] R_ECEF_BodyU = transposeMatr3(R_BodyU_ECEF);

        return R_ECEF_BodyU;
    }

    // Function that calculates the rotation matrix between Unity World reference frame and Unity Body reference frame, assuming that the Unity World reference frame is a topocentric one with its lat and lon
    public static double[,] BodyU_ECEF_MatR(double lat_deg, double lon_deg, double[] phiThetaPsi_Deg)
    {
        // ECEF_BODYU MATRIX, IT IS A 123 ROTATION OF PHI THETA PSI
        double[,] R_ECEF_BodyU = matRGeneral(1, 2, 3, phiThetaPsi_Deg[0] * DEG_TO_RAD, phiThetaPsi_Deg[1] * DEG_TO_RAD, phiThetaPsi_Deg[2] * DEG_TO_RAD);

        double[,] R_ECEF_ENU = ECEF_ENU_MatR(lat_deg, lon_deg);
        double[,] R_ENU_ECEF = transposeMatr3(R_ECEF_ENU);

        double[,] R_WorldU_ENU = new double[,] { { 1, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 } };

        // R_RANDOM = Matrix from the Unity body to the real body axes (front, right wing, down)
        //double[,] R_RANDOM = new double[,] { { 0, 1, 0 }, { 0, 0, -1 }, { 1, 0, 0 } }; // TO BE INSERTED

        // FINAL ROTATION MATRIX
        double[,] R_WorldU_BodyU = prodMatr3(R_ECEF_BodyU, prodMatr3(R_ENU_ECEF, R_WorldU_ENU));

        return R_WorldU_BodyU;
    }


    // ATTITUDE CALCULATION FROM ROTATION MATRICES FUNCTIONS

    // Get Tait Bryan angles for 213 (312?) rotation from Rotation matrix. MUST WORK FOR UNITY
    public static double[] TB_from_MatR_213(double[,] R_BodyU_ECEF)
    {
        // Psi, Theta, Phi vector
        double[] phiThetaPsi = new double[3];

        // Theta
        phiThetaPsi[0] = Math.Asin(-R_BodyU_ECEF[2, 1]);

        phiThetaPsi[1] = -Math.Atan2(R_BodyU_ECEF[2, 0], R_BodyU_ECEF[2, 2]);
        phiThetaPsi[2] = -Math.Atan2(R_BodyU_ECEF[0, 1], R_BodyU_ECEF[1, 1]);

        return phiThetaPsi;
    }

    // Get Tait Bryan angles for ANY rotation from Rotation matrix. 
    public static double[] TB_from_MatR(double[,] MatR, int[] axes)
    {
        // angles vector
        double[] angles = new double[3];

        if (axes[0] == 1 && axes[1] == 2 && axes[2] == 3)
        {
            angles[0] = Math.Atan2(-MatR[2, 1], MatR[2, 2]);
            angles[1] = Math.Asin(MatR[2, 0]);
            angles[2] = Math.Atan2(-MatR[1, 0], MatR[0, 0]);
        }
        else if (axes[0] == 3 && axes[1] == 2 && axes[2] == 1)
        {
            // Attention, angles 0 and 2 may be inverted
            angles[0] = Math.Atan2(MatR[0, 1], MatR[0, 0]);
            angles[1] = Math.Asin(-MatR[0, 2]);
            angles[2] = Math.Atan2(MatR[1, 2], MatR[2, 2]);
        }
        else if (axes[0] == 2 && axes[1] == 1 && axes[2] == 3)
        {
            angles[0] = Math.Asin(MatR[2, 1]);
            angles[1] = -Math.Atan2(MatR[2, 0], MatR[2, 2]);
            angles[2] = -Math.Atan2(MatR[0, 1], MatR[1, 1]);
        }
        else if (axes[0] == 3 && axes[1] == 1 && axes[2] == 3)
        {
            angles[0] = Math.Atan2(MatR[2, 0], -MatR[2, 1]);
            angles[1] = Math.Acos(MatR[2, 2]);
            angles[2] = Math.Atan2(MatR[0, 2], MatR[1, 2]);
        }
        else if (axes[0] == 9 && axes[1] == 9 && axes[2] == 9) // DBUG FOR VR-FORCES
        {
            angles[1] = Math.Asin(-MatR[0, 2]);
            angles[0] = (MatR[0, 1] >= 0.0) ? (Math.Acos(MatR[0, 0] / Math.Cos(angles[1]))) : (-Math.Acos(MatR[0, 0] / Math.Cos(angles[1])));
            angles[2] = (MatR[1, 2] >= 0.0) ? (Math.Acos(MatR[2, 2] / Math.Cos(angles[1]))) : (-Math.Acos(MatR[2, 2] / Math.Cos(angles[1])));
        }
        else
        {
            Debug.Log("WHAT ROTATION AXES DID YOU PROVIDE?");
        }
        return angles;
    }


    // ATTITUDE CONVERSION FROM ONE REF TO ANOTHER FUNCTIONS

    // Converts Tait-Bryan angles in Unity World reference (213 Tait bryan angles) to ECEF reference Tait bryan angles(phiThetaPsi (321 (123?)))
    public static double[] BodyU_ECEF_Attitude(double lat_deg, double lon_deg, double[] eulerAngles)
    {
        double[,] R_ECEF_BodyU = ECEF_BodyU_MatR(lat_deg, lon_deg, eulerAngles);
        double[] phiThetaPsi = TB_from_MatR(R_ECEF_BodyU, new int[] { 3, 2, 1 });

        return phiThetaPsi;
    }

    // Converts Tait-Bryan angles in Unity World reference (213 Tait bryan angles) to ECEF reference Tait bryan angles(phiThetaPsi (321 (123?)))
    public static double[] ECEF_BodyU_Attitude(double lat_deg, double lon_deg, double[] phiThetaPsi)
    {
        double[,] R_BodyU_ECEF = BodyU_ECEF_MatR(lat_deg, lon_deg, phiThetaPsi);
        double[] eulerAngles = TB_from_MatR_213(R_BodyU_ECEF);

        return eulerAngles;
    }

    // Converts Tait-Bryan angles in ECEF reference (TB_ECEF_deg) to Tait-Bryan angles in NED reference (TB_NED_deg)
    // Please note that for Psi and Phi (TB_NED_deg[0] and TB_NED_deg[2]) the values +180 and -180 are effectively equal
    public static double[] ECEF_NED_Attitude(double lat_deg, double lon_deg, double[] TB_ECEF_deg)
    {
        double[] TB_NED_rad, TB_NED_deg;

        double[,] rotMat_ECEF_NED, rotMat_ECEF_BODY;
        double[,] rotMat_NED_ECEF, rotMat_NED_BODY;

        // ECEF_BODY ROTATION MATRIX
        rotMat_ECEF_BODY = ECEF_BODY_MatR(TB_ECEF_deg);

        // ECEF_NED ROTATION MATRIX
        rotMat_ECEF_NED = ECEF_NED_MatR(lat_deg, lon_deg);
        rotMat_NED_ECEF = transposeMatr3(rotMat_ECEF_NED);

        // Third rotation matrix is NED to BODY
        // Calculated as matrix product of rotMat_ECEF_BODY by transpose of rotMat_ECEF_NED
        rotMat_NED_BODY = transposeMatr3(prodMatr3(rotMat_ECEF_BODY, rotMat_NED_ECEF));

        // From this rotation matrix I extract the angles Psi, Theta, Phi
        TB_NED_rad = TB_from_MatR(transposeMatr3(rotMat_NED_BODY), new int[] { 3, 2, 1 });
        // INVERTED ANGLES
        TB_NED_deg = new double[] { TB_NED_rad[2] / DEG_TO_RAD, TB_NED_rad[1] / DEG_TO_RAD, TB_NED_rad[0] / DEG_TO_RAD };

        return TB_NED_deg;
    }

    // Converts Tait-Bryan angles in NED reference (TB_NED_deg) to Tait-Bryan angles in ECEF reference (TB_ECEF_deg)
    // Please note that for TB_ECEF_deg[0] and TB_ECEF_deg[2] the values +180 and -180 are effectively equal
    public static double[] NED_ECEF_Attitude(double lat_deg, double lon_deg, double[] TB_NED_deg)
    {
        double[] TB_ECEF_rad, TB_ECEF_deg;

        double[,] rotMat_ECEF_NED, rotMat_NED_BODY;
        double[,] rotMat_ECEF_BODY;

        // ECEF_NED ROTATION MATRIX
        rotMat_ECEF_NED = ECEF_NED_MatR(lat_deg, lon_deg);

        // ECEF_BODY ROTATION MATRIX
        rotMat_NED_BODY = NED_BODY_MatR(TB_NED_deg);

        // Third rotation matrix is ECEF to BODY, Calculated as matrix product of rotMat_NED_BODY by rotMat_ECEF_NED
        rotMat_ECEF_BODY = prodMatr3(rotMat_NED_BODY, rotMat_ECEF_NED);

        // From this rotation matrix I extract the angles Alpha, Beta, Gamma
        TB_ECEF_rad = TB_from_MatR(rotMat_ECEF_BODY, new int[] {3, 2, 1});

        // Need to change sign for Alpha and Gamma
        TB_ECEF_deg = new double[] { TB_ECEF_rad[0] / DEG_TO_RAD, TB_ECEF_rad[1] / DEG_TO_RAD, TB_ECEF_rad[2] / DEG_TO_RAD };

        return TB_ECEF_deg;
    }
};

